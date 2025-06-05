import pandas as pd

ASAP_SCORES = {1: [2, 12], 2: [1, 6], 3: [0, 3], 4: [0, 3], 5: [0, 4], 6: [0, 4], 7: [0, 30], 8: [0, 60]}
ASAP_CLASSIFICATION_SCALE = [0, 1, 2, 3]


def normalize_data_asap(data, prompt):
    return (data - ASAP_SCORES.get(prompt)[0]) / (ASAP_SCORES.get(prompt)[1] - ASAP_SCORES.get(prompt)[0])


def rescale_data_asap(data, prompt):
    rescaled_data = ((data - ASAP_SCORES.get(prompt)[0]) / (ASAP_SCORES.get(prompt)[1] - ASAP_SCORES.get(prompt)[0]) * 4).astype(int)
    rescaled_data[rescaled_data > 3] = 3
    return rescaled_data


def preprocess_asap(file_name, p, setting):
    test_df = pd.read_csv(file_name + '.csv', encoding='utf-8')
    target = test_df['domain1_score']
    suffix = '_regression'
    if setting == 'regression':
        test_df['score'] = normalize_data_asap(target, p)
    elif setting == 'classification':
        test_df['score'] = rescale_data_asap(target, p)
        suffix = '_classification'
    else:
        print('Invalid Setting')
        exit(1)
    test_df = test_df.rename(columns={'essay_id': 'essay_id_comp', 'essay_set': 'prompt_name', 'essay': 'full_text',
                                'score': 'holistic_essay_score'})
    test_df.to_csv(file_name + suffix + '.csv', encoding='utf-8', index=False)
    return test_df


for setting in ['regression', 'classification']:
    print('***Setting ', setting, '***')
    for p in range(1, 9):
        print('***Prompt ', p, '***')
        test = preprocess_asap(str(p) + '/test', p, setting)
        #print(test.value_counts("holistic_essay_score"))
        train = preprocess_asap(str(p) + '/train', p, setting)
        print(train.value_counts("holistic_essay_score"))
        validation = preprocess_asap(str(p) + '/validation', p, setting)
        #print(validation.value_counts("holistic_essay_score"))
    test_all = pd.DataFrame()

    for p in range(1, 9):
        test = pd.read_csv(str(p) + '/test_'+setting+'.csv', encoding='utf-8')
        test_all = pd.concat([test_all, test])
    test_all.to_csv('test_all_'+setting+'.csv', encoding='utf-8', index=False)
