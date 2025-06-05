import pandas as pd


PERSUADE_SCORES = [1, 6]


def normalize_data_persuade(data):
    return (data - PERSUADE_SCORES[0]) / (PERSUADE_SCORES[1] - PERSUADE_SCORES[0])


def preprocess_persuade(file_name):
    test = pd.read_csv(file_name + '.csv', encoding='utf-8')
    target = test['holistic_essay_score']
    test['score'] = normalize_data_persuade(target)
    test = test.rename(columns={'essay_id_comp': 'essay_id_comp', 'prompt_name': 'prompt_name', 'full_text': 'full_text',
                                'holistic_essay_score':'domain1_score', 'score': 'holistic_essay_score'})
    test.to_csv(file_name.replace('classification','regression')+'.csv', encoding='utf-8', index=False)
    return test


for p in range(0, 15):
    print('***Prompt ', p, '***')
    test = preprocess_persuade(str(p) + '/test_classification')
    # print(test.value_counts("score"))
    train = preprocess_persuade(str(p) + '/train_classification')
    # print(train.value_counts("score"))
    validation = preprocess_persuade(str(p) + '/validation_classification')
    # print(validation.value_counts("score"))
test_all = pd.DataFrame()

for p in range(0, 15):
    test = pd.read_csv(str(p) + '/test_regression.csv', encoding='utf-8')
    test_all = pd.concat([test_all, test])

test_all.to_csv('test_all_regression.csv', encoding='utf-8', index=False)