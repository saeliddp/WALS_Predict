import csv

# last value in attribute_list must be what we try to predict
def collect_full_data(source_file, attribute_list):
    type_list = []
    for _ in attribute_list:
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
                # for debugging
                #lang_to_att[lang].append(lang)
                writer.writerow(lang_to_att[lang])
                num_written += 1
            elif lang_to_att[lang].index(None) == len(lang_to_att[lang]) - 1:
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
    collect_full_data('values.csv', ['1A', '4A', '11A', '12A'])
            