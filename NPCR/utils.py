import os
import sys
import logging

import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.metrics import classification_report, mean_squared_error, cohen_kappa_score


PROMPT_NAMES_TO_IDS = {
    'The Face on Mars': 0,
    'Facial action coding system': 1,
    "\"A Cowboy Who Rode the Waves\"": 2,
    "Does the electoral college work?": 3,
    "Car-free cities": 4,
    "Driverless cars": 5,
    "Exploring Venus": 6,
    "Summer projects": 7,
    "Mandatory extracurricular activities": 8,
    "Cell phones at school": 9,
    "Grades for extracurricular activities": 10,
    "Seeking multiple opinions": 11,
    "Phones and driving": 12,
    "Distance learning": 13,
    "Community service": 14
}
PROMPT_IDS_TEXT_DEPENDENT = [0, 1, 2, 3, 4, 5, 6]
PROMPT_IDS_INDEPENDENT = [7, 8, 9, 10, 11, 12, 13, 14]

ASAP_SCORES = {1: [2, 12], 2: [1, 6], 3: [0, 3], 4: [0, 3], 5: [0, 4], 6: [0, 4], 7: [0, 30], 8: [0, 60]}
ASAP_PROMPT_IDS_TEXT_DEPENDENT = [3, 4, 5, 6]
ASAP_PROMPT_IDS_INDEPENDENT = [1, 2, 7, 8]


def read_dataframe_asap(file_path):

    df = pd.read_csv(file_path)
    df = df.rename(columns={'holistic_essay_score': 'score_scaled', 'domain1_score': 'holistic_essay_score', 'prompt_name': 'prompt_id'})
    return df


def read_dataframe(file_path, col_prompt='prompt_name'):

    df = pd.read_csv(file_path)
    df['prompt_id'] = df[col_prompt].apply(lambda row: PROMPT_NAMES_TO_IDS[row])
    return df


def get_logger(name, level=logging.INFO, handler=sys.stdout,
        formatter='%(name)s - %(levelname)s - %(message)s'):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def rescale_tointscore_adversarial(scaled_scores, min_label=None, max_label=None, prompts_array=None, differences=False):

    if (min_label is None) and (max_label is None) and (prompts_array is None):
        logging.info('Cannot get model friendly scores: Neither min score, nor max score or prompts array was set!')
        sys.exit(0)

    if (min_label is not None) and (max_label is not None):
        # Scale to range of DEVIATION instead of pure scores
        if differences:
            min_label_copy = min_label
            max_label_copy = max_label
            min_label = min_label_copy - max_label_copy
            max_label = max_label_copy - min_label_copy
        # logging.info('Using min and max score to rescale labels! ' + str(min_label) + ' ' + str(max_label))
        int_scores = scaled_scores * (max_label - min_label) + min_label
        return int_scores
    
    else:
        # logging.info('Using prompt information to rescale labels! ' + str(prompts_array))    
        rescaled_scores = []
        for i in range(len(scaled_scores)):
            current_score = scaled_scores[i]
            current_prompt = prompts_array[i]
            min_label = ASAP_SCORES[current_prompt][0]
            max_label = ASAP_SCORES[current_prompt][1]
            # Scale to range of DEVIATION instead of pure scores
            if differences:
                min_label_copy = min_label
                max_label_copy = max_label
                min_label = min_label_copy - max_label_copy
                max_label = max_label_copy - min_label_copy
            rescaled_scores.append(current_score * (max_label - min_label) + min_label)
        
        return np.array(rescaled_scores)


# Write classification stats to file
def write_classification_stats(output_dir, y_true, y_pred, y_true_diff=None, y_pred_diff=None, suffix=''):

    with open(os.path.join(output_dir, 'stats' + suffix + '.csv'), 'w') as out_file:

        out_file.write(classification_report(y_true=y_true, y_pred=y_pred)+"\n\n")
        true_series = pd.Series(y_true, name='Actual')
        pred_series = pd.Series(y_pred, name='Predicted')
        out_file.write(str(pd.crosstab(true_series, pred_series))+"\n\n")
        out_file.write('QWK:\t' + str(cohen_kappa_score(y1 = y_true, y2 = y_pred, weights='quadratic')))
        pears_r, pears_p = pearsonr(y_true, y_pred)
        out_file.write('\nPearson:\t' + str(pears_r) + '\t(' + str(pears_p) + ')')

        # Partly for sanity (to compare against rmse in best validation epoch)
        if (y_true_diff is not None) and (y_pred_diff is not None):
            out_file.write('\n\nStats on raw predictions (on invididual reference examples):')
            out_file.write('\nRMSE:\t' + str(mean_squared_error(y_true = y_true_diff, y_pred = y_pred_diff, squared=False)))
            pearson_r, pearson_p = pearsonr(y_true_diff, y_pred_diff)
            out_file.write('\nPearson:\t' + str(pearson_r) + '\t(' + str(pearson_p) + ')')

            out_file.write('\nRMSE (smoothed):\t' + str(mean_squared_error(y_true = y_true_diff, y_pred = y_pred_diff, squared=False)))
            pearson_r, pearson_p = pearsonr(y_true_diff, y_pred_diff)
            out_file.write('\nPearson (smoothed):\t' + str(pearson_r) + '\t(' + str(pearson_p) + ')')
    
