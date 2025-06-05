import os, sys
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from copy import deepcopy
from utils import PROMPT_IDS_TEXT_DEPENDENT, PROMPT_IDS_INDEPENDENT, ASAP_PROMPT_IDS_TEXT_DEPENDENT, ASAP_PROMPT_IDS_INDEPENDENT


def get_within(row):

    training_prompt = row['training_prompt']
    if isinstance(training_prompt, str):
    # if '-' in training_prompt:
        training_prompts = training_prompt.split('-')
        training_prompts = [int(prompt) for prompt in training_prompts]
    else:
        training_prompts = [training_prompt]

    row_only_within = row[training_prompts]
    row_fisher = np.arctanh(list(row_only_within))
    avg_fisher = row_fisher.mean()
    return np.tanh(avg_fisher)


def get_cross(row, ref_prompts):

    ref_prompts = deepcopy(ref_prompts)
    training_prompt = row['training_prompt']
    if isinstance(training_prompt, str):
    # if '-' in training_prompt:
        training_prompts = training_prompt.split('-')
        training_prompts = [int(prompt) for prompt in training_prompts]
    else:
        training_prompts = [training_prompt]
    for training_prompt in training_prompts:
        if training_prompt in ref_prompts:
            ref_prompts.remove(training_prompt)
    row_only_cross = row[ref_prompts]
    row_fisher = np.arctanh(list(row_only_cross))
    avg_fisher = row_fisher.mean()
    return np.tanh(avg_fisher)


def get_avg_qwk(row, prompt_range=None):

    if prompt_range is not None:
        row = row[list(prompt_range)]
    row_fisher = np.arctanh(list(row))
    avg_fisher = np.mean(row_fisher)
    avg_qwk = np.tanh(avg_fisher)
    return avg_qwk
  

# From a folder of results, collect results for table
def analyze_baseline(data_path, prompts_range=range(15), dependent_prompts=PROMPT_IDS_TEXT_DEPENDENT, independent_prompts=PROMPT_IDS_INDEPENDENT):

    # To hold overall results: Train Prompt x Test Prompt
    result_dict = {}

    for train_prompt in prompts_range:

        prompt_folder = os.path.join(data_path, 'train_' + str(train_prompt))
        # For each test_prompt: put qwk
        qwk_dict = {}

        for test_prompt in prompts_range:

            df = pd.read_csv(os.path.join(prompt_folder, 'preds_' + str(test_prompt) + '.csv'))

            y_true = df.holistic_essay_score
            y_pred = df.pred
            qwk_dict[test_prompt] = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        
        result_dict[train_prompt] = qwk_dict
    

    df_results = pd.DataFrame.from_dict(result_dict)
    # Index represents training prompt, should be first column
    df_results.insert(0, 'training_prompt', df_results.index)

    # Look up within-prompt performance
    df_results['within'] = df_results.apply(lambda row: row[row['training_prompt']], axis='columns')
    all_prompts = list(prompts_range)
    # For different subgroups: Average evaluation on all prompts of the respective group
    df_results['cross_all'] = df_results.apply(get_cross, axis='columns', ref_prompts=all_prompts)
    df_results['cross_dependent'] = df_results.apply(get_cross, axis='columns', ref_prompts=dependent_prompts)
    df_results['cross_independent'] = df_results.apply(get_cross, axis='columns', ref_prompts=independent_prompts)
    df_results['avg'] = df_results.apply(get_avg_qwk, axis='columns', prompt_range=all_prompts)

    # Build averages over subsets of rows
    averages = []
    for targets in [dependent_prompts, independent_prompts, all_prompts]:
        avg = df_results.apply(get_avg_qwk, prompt_range=targets)
        avg['training_prompt'] = str(targets)
        averages.append(pd.DataFrame(avg).T)

    df_results = pd.concat([df_results] + averages)
    df_results = df_results.reset_index(drop=True)

    df_results.to_csv(os.path.join(data_path, 'result_overview.csv'), index=False)
    print(df_results)



def analyze_curves(results_path, prompt_range, dependent_prompts, independent_prompts):

    dict_results = {}
    dict_results_idx = 0

    dict_curve_results = {}
    dict_curve_results_idx = 0

    for prompt_type in ['integrated', 'independent']:

        for num_prompts_sampled in os.listdir(os.path.join(results_path, prompt_type)):

            num_prompts_sampled_int = int(num_prompts_sampled.split('_')[1])

            for sample_group in os.listdir(os.path.join(results_path, prompt_type, num_prompts_sampled)):

                for condition in ['adversarial', 'standard']:

                    current_dict = {'prompt_type': prompt_type, 'num_prompts': num_prompts_sampled_int, 'training_prompt': sample_group, 'training': condition}

                    for test_prompt in prompt_range:

                        df_preds = pd.read_csv(os.path.join(results_path, prompt_type, num_prompts_sampled, sample_group, condition, 'preds_' + str(test_prompt) + '.csv'))
                        qwk = cohen_kappa_score(df_preds['holistic_essay_score'], df_preds['pred'], weights='quadratic')
                        current_dict[test_prompt] = qwk

                    dict_results[dict_results_idx] = current_dict
                    dict_results_idx += 1
    
    df_results = pd.DataFrame.from_dict(dict_results, orient='index')
    all_prompts = list(prompt_range)
    df_results['cross_all'] = df_results.apply(get_cross, axis='columns', ref_prompts=all_prompts)
    df_results['cross_dependent'] = df_results.apply(get_cross, axis='columns', ref_prompts=dependent_prompts)
    df_results['cross_independent'] = df_results.apply(get_cross, axis='columns', ref_prompts=independent_prompts)
    df_results['avg_within'] = df_results.apply(get_within, axis='columns')

    df_results['avg_qwk'] = df_results.apply(get_avg_qwk, prompt_range=prompt_range, axis='columns')
    df_results.to_csv(os.path.join(results_path, 'raw_results.csv'))


    for num_prompts_sampled, df_num_sampled in df_results.groupby('num_prompts'):

        current_dict = {'num_mixed': num_prompts_sampled} 

        for prompt_type, df_prompt_type in df_num_sampled.groupby('prompt_type'):

            for condition, df_condition in df_prompt_type.groupby('training'):

                qwk_overall = get_avg_qwk(df_condition['avg_qwk'])
                qwk_cross_overall = get_avg_qwk(df_condition['cross_all'])
                qwk_cross_integrated = get_avg_qwk(df_condition['cross_dependent'])
                qwk_cross_independent = get_avg_qwk(df_condition['cross_independent'])
                qwk_within = get_avg_qwk(df_condition['avg_within'])
                current_dict[prompt_type + '_' + condition + '_overall'] = qwk_overall
                current_dict[prompt_type + '_' + condition + '_cross_overall'] = qwk_cross_overall
                current_dict[prompt_type + '_' + condition + '_cross_integrated'] = qwk_cross_integrated
                current_dict[prompt_type + '_' + condition + '_cross_independent'] = qwk_cross_independent
                current_dict[prompt_type + '_' + condition + '_within'] = qwk_within

        dict_curve_results[dict_curve_results_idx] = current_dict
        dict_curve_results_idx += 1 

    df_curve_results = pd.DataFrame.from_dict(dict_curve_results, orient='index')
    df_curve_results.to_csv(os.path.join(results_path, 'curve.csv'))


def analyze_curves_overall(results_path, prompt_range):

    dict_results = {}
    dict_results_idx = 0

    dict_curve_results = {}
    dict_curve_results_idx = 0

    for num_prompts_sampled in os.listdir(os.path.join(results_path)):
    
        if os.path.isdir(os.path.join(results_path, num_prompts_sampled)):
            num_prompts_sampled_int = int(num_prompts_sampled.split('_')[1])

            for sample_group in os.listdir(os.path.join(results_path, num_prompts_sampled)):

                if os.path.isdir(os.path.join(results_path, num_prompts_sampled, sample_group)):

                    for condition in ['adversarial', 'standard']:

                        current_dict = {'num_prompts': num_prompts_sampled_int, 'training_prompt': sample_group, 'training': condition}

                        for test_prompt in prompt_range:

                            df_preds = pd.read_csv(os.path.join(results_path, num_prompts_sampled, sample_group, condition, 'preds_' + str(test_prompt) + '.csv'))
                            qwk = cohen_kappa_score(df_preds['holistic_essay_score'], df_preds['pred'], weights='quadratic')
                            current_dict[test_prompt] = qwk

                        dict_results[dict_results_idx] = current_dict
                        dict_results_idx += 1
    
    df_results = pd.DataFrame.from_dict(dict_results, orient='index')
    all_prompts = list(prompt_range)
    df_results['cross_all'] = df_results.apply(get_cross, axis='columns', ref_prompts=all_prompts)
    df_results['avg_within'] = df_results.apply(get_within, axis='columns')

    df_results['avg_qwk'] = df_results.apply(get_avg_qwk, prompt_range=prompt_range, axis='columns')
    df_results.to_csv(os.path.join(results_path, 'raw_results.csv'))


    for num_prompts_sampled, df_num_sampled in df_results.groupby('num_prompts'):

        current_dict = {'num_mixed': num_prompts_sampled} 

        for condition, df_condition in df_num_sampled.groupby('training'):

            qwk_overall = get_avg_qwk(df_condition['avg_qwk'])
            qwk_cross_overall = get_avg_qwk(df_condition['cross_all'])
            qwk_within = get_avg_qwk(df_condition['avg_within'])
            current_dict[condition + '_overall'] = qwk_overall
            current_dict[condition + '_cross_overall'] = qwk_cross_overall
            current_dict[condition + '_within'] = qwk_within

        dict_curve_results[dict_curve_results_idx] = current_dict
        dict_curve_results_idx += 1 

    df_curve_results = pd.DataFrame.from_dict(dict_curve_results, orient='index')
    df_curve_results.to_csv(os.path.join(results_path, 'curve.csv'))




# Path to folder with prompt-wise results, where each folder contains individual predictions on all prompts
# analyze_baseline('/results/baseline/NPCR_longformer/PERSUADE')
# analyze_baseline('/results/baseline/NPCR_longformer/ASAP', prompts_range=range(1,9), dependent_prompts=ASAP_PROMPT_IDS_TEXT_DEPENDENT, independent_prompts=ASAP_PROMPT_IDS_INDEPENDENT)
# analyze_curves(results_path='results/curve/NPCR_longformer/ASAP', prompt_range=range(1,9), dependent_prompts=ASAP_PROMPT_IDS_TEXT_DEPENDENT, independent_prompts=ASAP_PROMPT_IDS_INDEPENDENT)
# analyze_curves(results_path='results/curve/NPCR_longformer/PERSUADE')
# analyze_curves_overall(results_path='results/curve/NPCR_longformer/ASAP_combined', prompt_range=range(1,9))
analyze_curves_overall(results_path='results/curve/NPCR_longformer/PERSUADE_combined', prompt_range=range(15))