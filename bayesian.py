import csv
from random import seed, randrange
from math import sqrt, pi, exp, ceil
# No class names can be 'length'
# first line of csv must describe whether each column is
# categorical or continuous
def load_csv(filename):
    rows = []
    with open(filename, newline='') as fr:
        for row in csv.reader(fr):
            rows.append(row)
    types = rows.pop(0)
    return [rows, types]

def str_cols_to_float(rows, cols):
    for row in rows:
        for col in cols:
            row[col] = float(row[col].strip())

def str_col_to_int(rows, column):
    class_values = [row[column] for row in rows]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup
    
def mean(data):
    try:
        return sum(data) / len(data)
    except:
        print(data)
        exit('Insufficient Data on this run. Mean not computable')
    
def stdev(data, avg=None):
    if avg is None:
        avg = mean(data)
    try:
        variance = sum((x - avg)**2 for x in data) / (len(data) - 1)
    except:
        print(data)
        exit('Insufficient Data on this run. Stdev not computable')
    return sqrt(variance)

# maps classes to all instances of that class
def separate_by_class(dataset):
    # assume last value is class def'n
    class_map = dict()
    for item in dataset:
        key = item[-1]
        if key not in class_map:
            class_map[key] = []
        class_map[key].append(item)
    return class_map

# preprocess the training set
def summarize_dataset(dataset, types):
    summaries = []
    index = 0
    for col in zip(*dataset):
        if types[index] == 'continuous':
            summaries.append((mean(col), stdev(col), len(col)))
        elif types[index] == 'discrete':
            freqs = dict()
            freqs['length'] = 0
            for val in col:
                if val == 'length':
                    exit('Error: attribute definition cannot be "length"')
                elif val in freqs:
                    freqs[val] += 1
                else:
                    freqs[val] = 1
                freqs['length'] += 1
            summaries.append(freqs)
        else:
            exit('Error: datatype definitions must be "continuous" or "discrete"')
        index += 1
            
    del(summaries[-1]) # get rid of class definition summary
    return summaries

def summarize_by_class(dataset, types):
    class_lists = separate_by_class(dataset)
    summaries = dict()
    for key in class_lists:
        summaries[key] = summarize_dataset(class_lists[key], types)
    #print(summaries)
    return summaries

# only relevant for continuous data
def gaussian_probability(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

def calculate_class_probabilities(new_row, class_summaries, types):
    num_training_rows = 0
    for key in class_summaries:
        if type(class_summaries[key][0]) == tuple:
            num_training_rows += class_summaries[key][0][-1]
        else:
            num_training_rows += class_summaries[key][0]['length']
    probs = dict()
    
    for key in class_summaries:
        try:
            prob = class_summaries[key][0][-1] / num_training_rows # P(class)
        except:
            prob = class_summaries[key][0]['length'] / num_training_rows
            
        for i, attribute_summary in enumerate(class_summaries[key]):
            if type(attribute_summary) == tuple:
                prob *= gaussian_probability(new_row[i], attribute_summary[0], attribute_summary[1])
            else:
                if new_row[i] in attribute_summary:
                    prob *= (attribute_summary[new_row[i]] / attribute_summary['length'])
                else: # no examples of this attribute value in training data
                    prob *= (1 / attribute_summary['length'])
    
        probs[key] = prob
    return probs

# predict class for a given row
def predict(row, summaries, types):
    probs = calculate_class_probabilities(row, summaries, types)
    best_class, best_value = None, -1
    for key in probs:
        if probs[key] > best_value:
            best_class = key
            best_value = probs[key]
    return best_class
    
def naive_bayes(train, test, types):
    summaries = summarize_by_class(train, types)
    predictions = []
    for row in test:
        predictions.append(predict(row, summaries, types))
    return predictions

def cross_validation_split(dataset, n_folds):
    dataset_copy = list(dataset)
    folds = []
    split_size = ceil(len(dataset) / n_folds)
    curr_fold_size = split_size
    while len(dataset_copy) > 0:
        if curr_fold_size >= split_size:
            folds.append([])
            curr_fold_size = 0
        folds[-1].append(dataset_copy.pop(randrange(len(dataset_copy))))
        curr_fold_size += 1
    
    return folds

def accuracy_metric(test_data, predicted_data):
    num_rows = 0
    num_successes = 0
    for i, row in enumerate(test_data):
        num_rows += 1
        if row[-1] == predicted_data[i]:
            num_successes += 1
    return (num_successes / num_rows) * 100

def evaluate_bayes(dataset, n_folds, types):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = naive_bayes(train_set, test_set, types)
		accuracy = accuracy_metric(fold, predicted)
		scores.append(accuracy)
	return scores
        
if __name__ == '__main__':
    #seed(1)
    filename = 'generated_data/complete_1A_4A_11A_12A.csv'
    data = load_csv(filename)
    dataset = data[0]
    types = data[1]
    for i, val in enumerate(types):
        if val == 'continuous':
            str_cols_to_float(dataset, [i])
    
    # single prediction
    """summaries = summarize_by_class(dataset, types)
    print(predict([1.3, 'red', None], summaries, types))
    """
    # convert class column to integers
    #str_col_to_int(dataset, len(dataset[0])-1)
    # evaluate algorithm
   
    n_folds = 5
    scores = evaluate_bayes(dataset, n_folds, types)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    
