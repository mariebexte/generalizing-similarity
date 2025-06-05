
import os
import sys
import random
import logging
import argparse

from datetime import datetime
import pandas as pd

from train_model import train_model
from utils import ASAP_PROMPT_IDS_TEXT_DEPENDENT, ASAP_PROMPT_IDS_INDEPENDENT, PROMPT_IDS_TEXT_DEPENDENT, PROMPT_IDS_INDEPENDENT, read_dataframe, read_dataframe_asap
from evaluator_core import evaluate_finetuned_model


INTEGRATED_SAMPLE={
    2:[(5, 6), (0, 4), (0, 1), (1, 4), (1, 3), (3, 5), (0, 3), (2, 3), (0, 2), (1, 6)],\
    3:[(0, 2, 5), (0, 1, 3), (1, 2, 5), (1, 2, 3), (4, 5, 6), (0, 1, 6), (1, 4, 6), (0, 1, 5), (1, 3, 6), (2, 4, 5)],\
    4:[(0, 1, 4, 5), (0, 1, 2, 4), (0, 3, 4, 6), (0, 2, 5, 6), (3, 4, 5, 6), (0, 1, 3, 4), (1, 2, 4, 5), (0, 1, 2, 6), (1, 2, 3, 5), (1, 3, 5, 6)],\
    5:[(2, 3, 4, 5, 6), (0, 1, 2, 4, 5), (0, 1, 2, 3, 4), (0, 1, 3, 5, 6), (0, 1, 3, 4, 6), (1, 2, 3, 4, 6), (0, 1, 2, 3, 6), (0, 2, 3, 4, 6), (0, 1, 2, 3, 5), (0, 2, 3, 4, 5)],\
    6:[(0, 1, 2, 3, 4, 5), (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6), (0, 1, 3, 4, 5, 6), (0, 1, 2, 4, 5, 6), (0, 1, 2, 3, 5, 6), (0, 1, 2, 3, 4, 6)],\
    7:[(0, 1, 2, 3, 4, 5, 6)]}

INDEPENDENT_SAMPLE={
    2: [(10, 13), (7, 11), (7, 8), (11, 13), (8, 10), (8, 9), (11, 12), (7, 12), (12, 14), (9, 14)],
    3: [(9, 11, 12), (7, 9, 11), (7, 8, 10), (10, 11, 13), (7, 11, 14), (7, 11, 12), (7, 10, 14), (7, 9, 12), (11, 12, 13), (7, 9, 10)],
    4: [(7, 8, 13, 14), (7, 8, 9, 13), (8, 9, 10, 11), (7, 11, 12, 13), (7, 10, 12, 13), (7, 9, 10, 13), (7, 8, 12, 14), (8, 9, 12, 14), (8, 10, 11, 14), (9, 10, 11, 14)],
    5: [(8, 9, 10, 13, 14), (7, 8, 9, 12, 13), (7, 8, 9, 10, 12), (8, 10, 11, 13, 14), (7, 8, 11, 12, 14), (7, 8, 10, 13, 14), (7, 8, 10, 12, 14), (7, 8, 9, 12, 14), (9, 10, 11, 13, 14), (7, 8, 9, 11, 14)],
    6: [(7, 10, 11, 12, 13, 14), (7, 8, 9, 10, 12, 13), (7, 8, 9, 10, 11, 12), (8, 9, 10, 11, 13, 14), (7, 8, 9, 11, 13, 14), (7, 8, 9, 11, 12, 14), (8, 9, 10, 11, 12, 14), (7, 8, 9, 10, 12, 14), (8, 10, 11, 12, 13, 14), (7, 9, 10, 11, 13, 14)],
    7: [(7, 8, 9, 10, 11, 12, 13), (8, 9, 10, 11, 12, 13, 14), (7, 9, 10, 11, 12, 13, 14), (7, 8, 10, 11, 12, 13, 14), (7, 8, 9, 11, 12, 13, 14), (7, 8, 9, 10, 12, 13, 14), (7, 8, 9, 10, 11, 13, 14), (7, 8, 9, 10, 11, 12, 14)],
    8: [(7, 8, 9, 10, 11, 12, 13, 14)]}


PERSUADE_FULL_SAMPLE={
    2: [(7, 12), (1, 2), (0, 4), (9, 14), (2, 11), (2, 7), (2, 4), (1, 5), (0, 14), (8, 11)],
    3: [(4, 10, 13), (0, 6, 9), (0, 1, 14), (6, 8, 10), (1, 6, 14), (1, 5, 7), (1, 4, 5), (0, 8, 10), (6, 7, 14), (0, 5, 12)],
    4: [(7, 8, 13, 14), (0, 4, 8, 14), (0, 1, 7, 9), (1, 5, 12, 13), (1, 4, 6, 14), (1, 3, 7, 8), (0, 6, 7, 13), (0, 4, 6, 8), (4, 7, 11, 14), (0, 3, 8, 9)],
    5: [(4, 5, 10, 12, 14), (0, 2, 7, 9, 10), (0, 1, 3, 8, 11), (1, 2, 5, 9, 14), (1, 2, 3, 4, 7), (0, 6, 9, 10, 14), (0, 3, 5, 8, 14), (0, 2, 5, 12, 13), (5, 6, 7, 13, 14), (3, 4, 5, 9, 11)],
    6: [(0, 2, 4, 5, 11, 14), (0, 1, 2, 9, 11, 12), (1, 2, 4, 8, 9, 11), (1, 2, 3, 4, 5, 10), (0, 5, 7, 9, 12, 13), (0, 2, 7, 8, 9, 13), (0, 2, 3, 7, 11, 12), (3, 6, 8, 10, 13, 14), (0, 1, 10, 11, 13, 14), (5, 6, 8, 9, 12, 13)],
    7: [(2, 4, 5, 7, 8, 9, 11), (0, 1, 4, 6, 7, 8, 12), (0, 1, 2, 4, 6, 7, 11), (3, 6, 9, 10, 12, 13, 14), (0, 3, 4, 7, 12, 13, 14), (0, 2, 6, 8, 11, 13, 14), (0, 2, 5, 6, 7, 8, 10), (0, 1, 5, 8, 11, 12, 14), (3, 6, 7, 8, 10, 11, 14), (0, 1, 4, 5, 6, 9, 11)],
    8: [(2, 3, 4, 5, 7, 12, 13, 14), (0, 1, 3, 4, 6, 9, 10, 11), (0, 1, 2, 3, 6, 7, 8, 9), (3, 4, 6, 7, 8, 9, 12, 14), (0, 2, 4, 5, 7, 9, 11, 12), (0, 2, 3, 5, 7, 9, 13, 14), (0, 2, 3, 4, 6, 8, 10, 13), (0, 1, 3, 6, 7, 9, 10, 11), (3, 4, 5, 7, 9, 10, 12, 13), (0, 1, 3, 4, 5, 7, 12, 14)],
    9: [(0, 1, 2, 7, 9, 10, 11, 12, 14), (0, 1, 2, 3, 4, 9, 12, 13, 14), (0, 2, 4, 5, 7, 8, 9, 11, 13), (0, 2, 3, 5, 6, 8, 9, 10, 14), (0, 2, 3, 4, 5, 9, 10, 11, 13), (0, 1, 3, 4, 7, 9, 10, 12, 13), (0, 1, 2, 5, 10, 11, 12, 13, 14), (2, 3, 4, 6, 9, 10, 11, 12, 14), (0, 1, 2, 4, 9, 11, 12, 13, 14), (3, 4, 5, 6, 9, 10, 11, 13, 14)],
    10: [(1, 3, 5, 6, 7, 8, 9, 10, 11, 14), (0, 1, 2, 3, 8, 9, 10, 11, 12, 14), (0, 1, 2, 3, 4, 5, 8, 10, 11, 13), (0, 1, 4, 5, 6, 7, 8, 9, 11, 12),  (0, 1, 3, 5, 6, 7, 8, 9, 10, 12), (0, 1, 3, 4, 5, 9, 10, 11, 13, 14), (0, 1, 2, 4, 5, 8, 9, 10, 12, 14), (0, 1, 2, 3, 6, 8, 9, 11, 12, 13), (2, 3, 4, 5, 6, 9, 11, 12, 13, 14), (1, 2, 3, 5, 6, 7, 9, 10, 11, 13)],
    11: [(2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14), (0, 1, 2, 3, 5, 6, 7, 8, 11, 13, 14), (0, 1, 2, 3, 4, 5, 6, 8, 11, 12, 13), (0, 1, 3, 4, 5, 7, 8, 10, 11, 13, 14), (0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 14), (0, 1, 2, 5, 6, 7, 8, 9, 11, 12, 14), (0, 1, 2, 3, 5, 7, 9, 11, 12, 13, 14), (0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14), (1, 2, 3, 4, 7, 8, 9, 10, 12, 13, 14), (0, 1, 2, 3, 4, 6, 9, 10, 11, 13, 14)],
    12: [(0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14), (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13), (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 14), (1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13), (0, 1, 2, 3, 5, 6, 7, 10, 11, 12, 13, 14), (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 13, 14), (0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 13, 14), (0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13), (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14), (0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 13, 14)],
    13: [(0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14), (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13), (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14), (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14), (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14), (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13), (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14), (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14), (0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14)],
    14: [(0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14), (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14), (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14), (0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)],
    15: [(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)]
}




ASAP_INTEGRATED_SAMPLE={
    2: [(3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6)],
    3: [(3, 4, 5), (3, 4, 6), (3, 5, 6), (4, 5, 6)],
    4: [(3, 4, 5, 6)]
}

ASAP_INDEPENDENT_SAMPLE={
    2: [(1, 2), (1, 7), (1, 8), (2, 7), (2, 8), (7, 8)],
    3: [(1, 2, 7), (1, 2, 8), (1, 7, 8), (2, 7, 8)],
    4: [(1, 2, 7, 8)]
}

ASAP_FULL_SAMPLE={
    2: [(4, 7), (1, 5), (1, 2), (5, 7), (2, 4), (2, 3), (5, 6), (1, 6), (6, 8), (3, 8)],
    3: [(3, 5, 6), (1, 3, 5), (1, 2, 4), (4, 5, 7), (1, 5, 8), (1, 5, 6), (1, 4, 8), (1, 3, 6), (5, 6, 7), (1, 3, 4)],
    4: [(1, 2, 7, 8), (1, 2, 3, 7), (2, 3, 4, 5), (1, 5, 6, 7), (1, 4, 6, 7), (1, 3, 4, 7), (1, 2, 6, 8), (2, 3, 6, 8), (2, 4, 5, 8), (3, 4, 5, 8)],
    5: [(2, 3, 4, 7, 8), (1, 2, 3, 6, 7), (1, 2, 3, 4, 6), (2, 4, 5, 7, 8), (1, 2, 5, 6, 7), (1, 2, 4, 7, 8), (1, 2, 4, 6, 8), (1, 2, 3, 6, 8), (3, 4, 5, 7, 8), (1, 2, 3, 5, 8)],
    6: [(1, 4, 5, 6, 7, 8), (1, 2, 3, 5, 6, 7), (1, 2, 3, 4, 5, 6), (2, 3, 4, 5, 7, 8), (1, 2, 3, 5, 7, 8), (1, 2, 3, 5, 6, 8), (2, 3, 4, 5, 6, 8), (1, 2, 3, 4, 6, 8), (2, 4, 5, 6, 7, 8), (1, 3, 4, 5, 7, 8)],
    7: [(1, 2, 3, 4, 5, 6, 7), (2, 3, 4, 5, 6, 7, 8), (1, 3, 4, 5, 6, 7, 8), (1, 2, 4, 5, 6, 7, 8), (1, 2, 3, 5, 6, 7, 8), (1, 2, 3, 4, 6, 7, 8), (1, 2, 3, 4, 5, 7, 8), (1, 2, 3, 4, 5, 6, 8)],
    8: [(1, 2, 3, 4, 5, 6, 7, 8)]
}

ASAP_SAVE_MODEL_COMBINATIONS=[(1, 2, 3, 4, 5, 6, 7, 8), (1, 2, 3, 5, 6, 7), (1, 2, 5, 6, 7), (1, 5, 6, 7), (1, 5, 6), (1, 5)]


def get_mixed_data(prompts, target_amount, filename, data_path='../data', persuade=True):

    random.seed(42)
    amount_per_prompt = int(target_amount/len(prompts))

    dfs_sampled = []
    for prompt in prompts:
        if persuade:
            prompt_df = read_dataframe(os.path.join(data_path, str(prompt), filename))
        else:
            prompt_df = read_dataframe_asap(os.path.join(data_path, str(prompt), filename))
        sample_ids = random.sample(list(prompt_df['essay_id_comp'].unique()), amount_per_prompt)
        sample_df = prompt_df.loc[prompt_df['essay_id_comp'].isin(sample_ids)]
        dfs_sampled.append(sample_df)
    combi_df = pd.concat(dfs_sampled)
    combi_df = combi_df.reset_index(drop=True)

    return combi_df


def train_and_eval_on_all_prompts(target_path, df_train, df_val, data_path, model_name='best_model.pt', max_num=1024, training_with_same_score=False, adversarial=False, range_prompts=range(15), num_training_pairs=None, min_label=None, max_label=None, persuade=True, keep_model=False):

    if adversarial:
        training_within_prompt=False
    else:
        training_within_prompt=True

    train_model(target_path=target_path, df_train=df_train, df_val=df_val, df_test=None, model_name=model_name, max_num=max_num, min_label=min_label, max_label=max_label, training_within_prompt=training_within_prompt, training_with_same_score=training_with_same_score, num_training_pairs=num_training_pairs)

    # Evaluate on all prompts
    for test_prompt in range_prompts:
        if persuade:
            df_test = read_dataframe(os.path.join(data_path, str(test_prompt), 'test.csv'))
        else:
            df_test = read_dataframe_asap(os.path.join(data_path, str(test_prompt), 'test_scaled.csv'))
        evaluate_finetuned_model(model_path=os.path.join(target_path, model_name), df_ref=df_train, df_test=df_test, target_path=target_path, max_num=max_num, suffix='_' + str(test_prompt), min_label=min_label, max_label=max_label)

    # Delete model
    if not keep_model:
        if os.path.exists(os.path.join(target_path, model_name)):
            os.remove(os.path.join(target_path, model_name))


def run_curve_persuade(out_dir='/results', target_folder='curve/NPCR_longformer/PERSUADE', integrated_prompts=False, training_with_same_score=False, adversarial=False):

    num_training_pairs = 1495
    if training_with_same_score:
        num_training_pairs = 2325
    
    training_prompts = []
    prompt_id_sample_dict = None

    if integrated_prompts:
        training_prompts = PROMPT_IDS_TEXT_DEPENDENT
        prompt_id_sample_dict = INTEGRATED_SAMPLE
    else:
        training_prompts = PROMPT_IDS_INDEPENDENT
        prompt_id_sample_dict = INDEPENDENT_SAMPLE

    for num_prompts_to_sample, prompt_ids_to_sample in prompt_id_sample_dict.items():

        print('Starting with combinations of ' + str(num_prompts_to_sample)+ ' prompts!')

        for id_tuple in prompt_ids_to_sample:

            condition = 'independent'
            if integrated_prompts:
                condition = 'integrated'

            subdir = 'standard'
            if adversarial:
                subdir = 'adversarial'

            id_list = [str(element) for element in id_tuple]
            target_path = os.path.join(out_dir, target_folder, condition, 'sample_' + str(num_prompts_to_sample), '-'.join(id_list), subdir)

            if not os.path.exists(target_path):
                os.makedirs(target_path)

            # Clear loggers from previous runs
            logger = logging.getLogger()
            for handler in logger.handlers[:]:
                handler.stream.close()
                logger.removeHandler(handler)
            logging.basicConfig(filename=os.path.join(target_path, datetime.now().strftime('logs_%H_%M_%d_%m_%Y.log')), filemode='w', level=logging.INFO)

            logging.info(f"Running adversarial: {adversarial}")
            logging.info(f"Running integrated prompts: {integrated_prompts}")
            logging.info('Running combination of ' + str(num_prompts_to_sample) + ' prompts, specifically IDs ' + str(id_tuple))

            df_train = get_mixed_data(prompts=id_tuple, target_amount=800, filename='train_800.csv')
            df_val = get_mixed_data(prompts=id_tuple, target_amount=100, filename='validation.csv')

            logging.info('Number of training texts:' + str(len(df_train)))
            logging.info('Number of validation texts:' + str(len(df_val)))

            if not os.path.exists(os.path.join(target_path, 'stats_0.csv')):
                train_and_eval_on_all_prompts(target_path=target_path, df_train=df_train, df_val=df_val, data_path='../data', training_with_same_score=training_with_same_score, adversarial=adversarial, num_training_pairs=num_training_pairs, min_label=1, max_label=6)


def run_curve_asap(out_dir='/results', target_folder='curve/NPCR_longformer/ASAP', integrated_prompts=False, training_with_same_score=False, adversarial=False):

    num_training_pairs = 945
    if training_with_same_score:
        num_training_pairs = 1610
    
    training_prompts = []
    prompt_id_sample_dict = None

    if integrated_prompts:
        training_prompts = ASAP_PROMPT_IDS_TEXT_DEPENDENT
        prompt_id_sample_dict = ASAP_INTEGRATED_SAMPLE
    else:
        training_prompts = ASAP_PROMPT_IDS_INDEPENDENT
        prompt_id_sample_dict = ASAP_INDEPENDENT_SAMPLE

    for num_prompts_to_sample, prompt_ids_to_sample in prompt_id_sample_dict.items():

        print('Starting with combinations of ' + str(num_prompts_to_sample)+ ' prompts!')

        for id_tuple in prompt_ids_to_sample:

            condition = 'independent'
            if integrated_prompts:
                condition = 'integrated'

            subdir = 'standard'
            if adversarial:
                subdir = 'adversarial'

            id_list = [str(element) for element in id_tuple]
            target_path = os.path.join(out_dir, target_folder, condition, 'sample_' + str(num_prompts_to_sample), '-'.join(id_list), subdir)

            if not os.path.exists(target_path):
                os.makedirs(target_path)

            # Clear loggers from previous runs
            logger = logging.getLogger()
            for handler in logger.handlers[:]:
                handler.stream.close()
                logger.removeHandler(handler)
            logging.basicConfig(filename=os.path.join(target_path, datetime.now().strftime('logs_%H_%M_%d_%m_%Y.log')), filemode='w', level=logging.INFO)

            logging.info(f"Running adversarial: {adversarial}")
            logging.info(f"Running integrated prompts: {integrated_prompts}")
            logging.info('Running combination of ' + str(num_prompts_to_sample) + ' prompts, specifically IDs ' + str(id_tuple))

            df_train = get_mixed_data(prompts=id_tuple, target_amount=560, data_path='../asap', filename='train_scaled.csv', persuade=False)
            df_val = get_mixed_data(prompts=id_tuple, target_amount=70, data_path='../asap', filename='validation_scaled.csv', persuade=False)

            logging.info('Number of training texts:' + str(len(df_train)))
            logging.info('Number of validation texts:' + str(len(df_val)))

            if not os.path.exists(os.path.join(target_path, 'stats_0.csv')):
                train_and_eval_on_all_prompts(target_path=target_path, df_train=df_train, df_val=df_val, data_path = '../asap', training_with_same_score=training_with_same_score, adversarial=adversarial, num_training_pairs=num_training_pairs, range_prompts=range(1,9), persuade=False)



def run_curve_asap_combined(combinations, save_combinations=ASAP_SAVE_MODEL_COMBINATIONS, out_dir='/results', target_folder='curve/NPCR_longformer/ASAP_combined', training_with_same_score=False, adversarial=False):

    num_training_pairs = 920
    if training_with_same_score:
        num_training_pairs = 1605

    for id_tuple in combinations:


        num_prompts_to_sample = len(id_tuple)

        keep_model = False
        if id_tuple in save_combinations:
            keep_model = True

        subdir = 'standard'
        if adversarial:
            subdir = 'adversarial'

        id_list = [str(element) for element in id_tuple]
        target_path = os.path.join(out_dir, target_folder, 'sample_' + str(num_prompts_to_sample), '-'.join(id_list), subdir)

        if not os.path.exists(target_path):
            os.makedirs(target_path)

        # Clear loggers from previous runs
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            handler.stream.close()
            logger.removeHandler(handler)
        logging.basicConfig(filename=os.path.join(target_path, datetime.now().strftime('logs_%H_%M_%d_%m_%Y.log')), filemode='w', level=logging.INFO)

        logging.info(f"Running adversarial: {adversarial}")
        logging.info('Running combination of ' + str(num_prompts_to_sample) + ' prompts, specifically IDs ' + str(id_tuple))

        df_train = get_mixed_data(prompts=id_tuple, target_amount=560, data_path='../asap', filename='train_scaled.csv', persuade=False)
        df_val = get_mixed_data(prompts=id_tuple, target_amount=70, data_path='../asap', filename='validation_scaled.csv', persuade=False)

        logging.info('Number of training texts:' + str(len(df_train)))
        logging.info('Number of validation texts:' + str(len(df_val)))

        if not os.path.exists(os.path.join(target_path, 'stats_0.csv')):
            train_and_eval_on_all_prompts(target_path=target_path, df_train=df_train, df_val=df_val, data_path = '../asap', training_with_same_score=training_with_same_score, adversarial=adversarial, num_training_pairs=num_training_pairs, range_prompts=range(1,9), persuade=False, keep_model=keep_model)


def run_curve_persuade_combined(combinations, out_dir='/results', target_folder='curve/NPCR_longformer/PERSUADE_combined', training_with_same_score=False, adversarial=False):

    num_training_pairs = 1474
    if training_with_same_score:
        num_training_pairs = 2250

    for id_tuple in combinations:

        num_prompts_to_sample = len(id_tuple)

        subdir = 'standard'
        if adversarial:
            subdir = 'adversarial'

        id_list = [str(element) for element in id_tuple]
        target_path = os.path.join(out_dir, target_folder, 'sample_' + str(num_prompts_to_sample), '-'.join(id_list), subdir)

        if not os.path.exists(target_path):
            os.makedirs(target_path)

        # Clear loggers from previous runs
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            handler.stream.close()
            logger.removeHandler(handler)
        logging.basicConfig(filename=os.path.join(target_path, datetime.now().strftime('logs_%H_%M_%d_%m_%Y.log')), filemode='w', level=logging.INFO)

        logging.info(f"Running adversarial: {adversarial}")
        logging.info('Running combination of ' + str(num_prompts_to_sample) + ' prompts, specifically IDs ' + str(id_tuple))

        df_train = get_mixed_data(prompts=id_tuple, target_amount=800, filename='train_800.csv')
        df_val = get_mixed_data(prompts=id_tuple, target_amount=100, filename='validation.csv')

        logging.info('Number of training texts:' + str(len(df_train)))
        logging.info('Number of validation texts:' + str(len(df_val)))

        if not os.path.exists(os.path.join(target_path, 'stats_0.csv')):
            train_and_eval_on_all_prompts(target_path=target_path, df_train=df_train, df_val=df_val, data_path='../data', training_with_same_score=training_with_same_score, adversarial=adversarial, num_training_pairs=num_training_pairs, min_label=1, max_label=6)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--out_dir')

    parser.add_argument('--adversarial', action='store_true')
    parser.add_argument('--no-adversarial', dest='feature', action='store_false')

    parser.add_argument('--integrated', action='store_true')
    parser.add_argument('--no-integrated', dest='feature', action='store_false')

    parser.add_argument('--persuade', action='store_true')
    parser.add_argument('--asap', action='store_true')
    parser.add_argument('--asap_combined', action='store_true')
    parser.add_argument('--persuade_combined', action='store_true')

    parser.add_argument('--first_half', action='store_true')
    parser.add_argument('--second_half', action='store_true')


    args = parser.parse_args()
    adversarial = args.adversarial
    integrated = args.integrated

    out_dir=args.out_dir

    persuade = args.persuade
    asap = args.asap
    asap_combined = args.asap_combined
    persuade_combined = args.persuade_combined

    first_half = args.first_half
    second_half = args.second_half

    combinations = []
    if first_half:
        for num_paired in [2, 3, 4, 5, 6, 7, 8]:
            combinations = combinations + PERSUADE_FULL_SAMPLE[num_paired]
    if second_half:
        for num_paired in [9, 10, 11, 12, 13, 14, 15]:
            combinations = combinations + PERSUADE_FULL_SAMPLE[num_paired]

    if asap:
        run_curve_asap(out_dir=out_dir, integrated_prompts=integrated, adversarial=adversarial)

    if asap_combined:
        run_curve_asap_combined(combinations=combinations, out_dir=out_dir, adversarial=adversarial)

    if persuade:
        run_curve_persuade(out_dir=out_dir, integrated_prompts=integrated, adversarial=adversarial)

    if persuade_combined:
        run_curve_persuade_combined(combinations=combinations, out_dir=out_dir, adversarial=adversarial)
