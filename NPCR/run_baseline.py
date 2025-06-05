import os
import sys
import logging

from utils import read_dataframe, read_dataframe_asap, ASAP_SCORES
from datetime import datetime
from train_model import train_model
from evaluator_core import evaluate_finetuned_model


def run_baseline_persuade(data_path='../data', training_within_prompt=True, training_with_same_score=False, do_longformer=True, min_label=1, max_label=6):

    max_num = 512
    if do_longformer:
        max_num = 1024

    for prompt in range(15):

        model = ''
        if do_longformer:
            model = '_longformer'

        num_training_pairs = 1495
        if training_with_same_score:
            num_training_pairs = 2325
            model = model + '_same_score'
        
        target_path = os.path.join('/results/baseline/NPCR' + model, 'PERSUADE', 'train_' + str(prompt))
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        # Clear loggers from previous runs
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            handler.stream.close()
            logger.removeHandler(handler)
        logging.basicConfig(filename=os.path.join(target_path, datetime.now().strftime('logs_%H_%M_%d_%m_%Y.log')), filemode='w', level=logging.INFO)

        start = datetime.now()

        model_name = 'best_model_prompt_' + str(prompt) + '.pt'

        train_path = os.path.join(data_path, str(prompt), 'train_800.csv')
        val_path = os.path.join(data_path, str(prompt), 'validation.csv')
        test_path = os.path.join(data_path, str(prompt), 'test.csv')

        logging.info('training data path:\t' + train_path)
        logging.info('validation data path:\t' + val_path)
        logging.info('test data path:\t' + test_path)

        df_train = read_dataframe(train_path)
        df_val = read_dataframe(val_path)
        df_test = read_dataframe(test_path)

        # Train a model, but only if this prompt did not already run
        if not os.path.exists(os.path.join(target_path, 'stats.csv')):
            if do_longformer:
                train_model(target_path=target_path, df_train=df_train, df_val=df_val, df_test=df_test, max_num=max_num, model_name=model_name, longformer=do_longformer, training_within_prompt=training_within_prompt, training_with_same_score=training_with_same_score, num_training_pairs=num_training_pairs, min_label=min_label, max_label=max_label)
            else:
                train_model(target_path=target_path, df_train=df_train, df_val=df_val, df_test=df_test, max_num=max_num, num_epochs=40, model_name=model_name, longformer=do_longformer, training_within_prompt=training_within_prompt, training_with_same_score=training_with_same_score, num_training_pairs=num_training_pairs, min_label=min_label, max_label=max_label)


            # Evaluate on all prompts separately
            for test_prompt in range(15):
                logging.info('START EVAL ON PROMPT ' + str(test_prompt))
                other_test_path = os.path.join(data_path, str(test_prompt), 'test.csv')
                df_test_other = read_dataframe(other_test_path)
                logging.info('load test data from:\t' + other_test_path)
                evaluate_finetuned_model(model_path=os.path.join(target_path, model_name), df_ref=df_train, df_test=df_test_other, target_path=target_path, max_num=max_num, suffix='_' + str(test_prompt), min_label=min_label, max_label=max_label)

            # Delete model
            if os.path.exists(os.path.join(target_path, model_name)):
                os.remove(os.path.join(target_path, model_name))

            logging.info('Everything (including running test on all prompts) took:\t' + str(datetime.now() - start))

        else:
            print('Skipping prompt ' + str(prompt) + ' because it already ran!')



def run_baseline_asap(data_path='../asap', training_within_prompt=True, training_with_same_score=False, do_longformer=True):

    max_num = 512
    if do_longformer:
        max_num = 1024

    for prompt in range(1, 9):

        model = ''
        if do_longformer:
            model = '_longformer'

        num_training_pairs = 920
        if training_with_same_score:
            num_training_pairs = 1605
            model = model + '_same_score'
        
        target_path = os.path.join('/results/baseline/NPCR' + model, 'ASAP', 'train_' + str(prompt))
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        # Clear loggers from previous runs
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            handler.stream.close()
            logger.removeHandler(handler)
        logging.basicConfig(filename=os.path.join(target_path, datetime.now().strftime('logs_%H_%M_%d_%m_%Y.log')), filemode='w', level=logging.INFO)

        start = datetime.now()

        model_name = 'best_model_prompt_' + str(prompt) + '.pt'

        train_path = os.path.join(data_path, str(prompt), 'train_scaled.csv')
        val_path = os.path.join(data_path, str(prompt), 'validation_scaled.csv')
        test_path = os.path.join(data_path, str(prompt), 'test_scaled.csv')

        logging.info('training data path:\t' + train_path)
        logging.info('validation data path:\t' + val_path)
        logging.info('test data path:\t' + test_path)

        df_train = read_dataframe_asap(train_path)
        df_val = read_dataframe_asap(val_path)
        df_test = read_dataframe_asap(test_path)

        # Train a model, but only if this prompt did not already run
        if not os.path.exists(os.path.join(target_path, 'stats.csv')):
            if do_longformer:
                train_model(target_path=target_path, df_train=df_train, df_val=df_val, df_test=df_test, max_num=max_num, model_name=model_name, longformer=do_longformer, training_within_prompt=training_within_prompt, training_with_same_score=training_with_same_score)
            else:
                train_model(target_path=target_path, df_train=df_train, df_val=df_val, df_test=df_test, max_num=max_num, num_epochs=40, model_name=model_name, longformer=do_longformer, training_within_prompt=training_within_prompt, training_with_same_score=training_with_same_score)


            # Evaluate on all prompts separately
            for test_prompt in range(1, 9):
                logging.info('START EVAL ON PROMPT ' + str(test_prompt))
                other_test_path = os.path.join(data_path, str(test_prompt), 'test_scaled.csv')
                df_test_other = read_dataframe_asap(other_test_path)
                logging.info('load test data from:\t' + other_test_path)
                evaluate_finetuned_model(model_path=os.path.join(target_path, model_name), df_ref=df_train, df_test=df_test_other, target_path=target_path, max_num=max_num, suffix='_' + str(test_prompt))

            # Delete model
            if os.path.exists(os.path.join(target_path, model_name)):
                os.remove(os.path.join(target_path, model_name))

            logging.info('Everything (including running test on all prompts) took:\t' + str(datetime.now() - start))

        else:
            print('Skipping prompt ' + str(prompt) + ' because it already ran!')



## PERSUADE
# run_baseline_persuade()

## ASAP
run_baseline_asap()