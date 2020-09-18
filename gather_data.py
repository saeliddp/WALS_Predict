import csv

language_list = []
def get_location(language_code):
    if len(language_list) == 0:
        with open('languages.csv', newline='') as fr:
            for line in csv.reader(fr):
                language_list.append(line)
                
    target_row = binary_search(language_code, language_list)
    if target_row is not None:
        return [target_row[3], target_row[4]]

def binary_search(lang_code, lang_list):
    return bs_helper(lang_code, lang_list, 0, len(lang_list) - 1)

def bs_helper(lang_code, lang_list, start, end):
    mid = int((start + end) / 2)
    curr_code = lang_list[mid][0]
    if curr_code == lang_code:
        return lang_list[mid]
    elif end - start <= 1:
        return None
    elif curr_code > lang_code:
        return bs_helper(lang_code, lang_list, start, mid)
    else:
        return bs_helper(lang_code, lang_list, mid, end)
        
# last value in attribute_list must be what we try to predict
def collect_full_data(source_file, attribute_list, lat=False, long=False):
    type_list = []
    for _ in attribute_list:
        type_list.append('discrete')
    if lat:
        type_list[-1] = 'continuous'
        type_list.append('discrete')
    if long:
        type_list[-1] = 'continuous'
        type_list.append('discrete')
        
    lang_to_att = dict()
    with open(source_file, newline='', encoding='utf-8') as fr:
        for line in csv.reader(fr):
            if line[2] in attribute_list:
                if line[1] not in lang_to_att:
                    lang_to_att[line[1]] = []
                    for _ in attribute_list:
                        lang_to_att[line[1]].append(None)
                target_index = attribute_list.index(line[2])
                lang_to_att[line[1]][target_index] = line[3]
    
    missing_only_output = []
    output_prefix = 'generated_data/'
    output_suffix = '_'.join(attribute_list) + '.csv'
    num_written = 0
    with open(output_prefix + 'complete_' + output_suffix, 'w', newline='') as fw:
        writer = csv.writer(fw)
        writer.writerow(type_list)
        for lang in lang_to_att:
            if None not in lang_to_att[lang]:
                if lat or long:
                    location = get_location(lang)
                    if lat:
                        temp = lang_to_att[lang][-1]
                        lang_to_att[lang][-1] = location[0]
                        lang_to_att[lang].append(temp)
                    if long:
                        temp = lang_to_att[lang][-1]
                        lang_to_att[lang][-1] = location[1]
                        lang_to_att[lang].append(temp)
                # for debugging
                #lang_to_att[lang].append(lang)
                writer.writerow(lang_to_att[lang])
                num_written += 1
            elif lang_to_att[lang].index(None) == len(lang_to_att[lang]) - 1:
                if lat or long:
                    location = get_location(lang)
                    if lat:
                        temp = lang_to_att[lang][-1]
                        lang_to_att[lang][-1] = location[0]
                        lang_to_att[lang].append(temp)
                    if long:
                        temp = lang_to_att[lang][-1]
                        lang_to_att[lang][-1] = location[1]
                        lang_to_att[lang].append(temp)
                lang_to_att[lang].append(lang)
                missing_only_output.append(lang_to_att[lang])
    
    with open(output_prefix + 'incomplete_' + output_suffix, 'w', newline='') as fw:
        writer = csv.writer(fw)
        writer.writerow(type_list)
        for line in missing_only_output:
            writer.writerow(line)
    
    print(str(len(lang_to_att)) + ' languages represented.')
    print(str(num_written) + ' languages have complete data.')
    print(str(len(missing_only_output)) + ' languages missing only output variable.')
    
if __name__ == '__main__':
    collect_full_data('values.csv', ['13A', '14A', '9A'], lat=True, long=True)
            