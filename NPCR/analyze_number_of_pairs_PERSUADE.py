import os
import data_prepare

from utils import read_dataframe
from run_curve import INDEPENDENT_SAMPLE, INTEGRATED_SAMPLE, PERSUADE_FULL_SAMPLE, get_mixed_data

# pair trainig data within-prompt
with_same_score_within = []
without_same_score_within = []

# pair training data cross-prompt
with_same_score_across = []
without_same_score_across = []

# single prompts
for prompt in range(15):

    df = read_dataframe(os.path.join('../data', str(prompt), 'train_800.csv'))
    df_train, df_val, df_test = data_prepare.prepare_sentence_data_adversarial(df_train=df, df_val=None, df_test=None, max_num=1024)

    # build pairs with same score
    features_train_same_score, _, _ = data_prepare.get_training_pairs(df_train=df_train, training_within_prompt=True, training_with_same_score=True, min_label=1, max_label=6)

    # build pairs without same score
    features_train, _, _ = data_prepare.get_training_pairs(df_train=df_train, training_within_prompt=True, training_with_same_score=False, min_label=1, max_label=6)

    with_same_score_within.append(len(features_train_same_score))
    without_same_score_within.append(len(features_train))


# mixed prompts
# for prompt_info in [INDEPENDENT_SAMPLE, INTEGRATED_SAMPLE]:
for prompt_info in [PERSUADE_FULL_SAMPLE]:

    for number_of_prompts, prompt_ids in prompt_info.items():

        for prompt_id_element in prompt_ids:
            df_train_mixed = get_mixed_data(prompts=prompt_id_element, target_amount=800, filename='train_800.csv', data_path='../data')
            df_train_mixed, df_val, df_test = data_prepare.prepare_sentence_data_adversarial(df_train=df_train_mixed, df_val=None, df_test=None, max_num=1024)

            # Normal: Training pairs within prompt
            # build pairs with same score
            feat_train_within_same, _, _ = data_prepare.get_training_pairs(df_train=df_train_mixed, training_within_prompt=True, training_with_same_score=True, min_label=1, max_label=6)
            # build pairs without same score
            feat_train_within, _, _ = data_prepare.get_training_pairs(df_train=df_train_mixed, training_within_prompt=True, training_with_same_score=False, min_label=1, max_label=6)

            # Adversarial
            # build pairs with same score
            feat_train_adversarial_same, _, _ = data_prepare.get_training_pairs(df_train=df_train_mixed, training_within_prompt=False, training_with_same_score=True, min_label=1, max_label=6)
            # build pairs without same score
            feat_train_adversarial, _, _ = data_prepare.get_training_pairs(df_train=df_train_mixed, training_within_prompt=False, training_with_same_score=False, min_label=1, max_label=6)

            with_same_score_within.append(len(feat_train_within_same))
            without_same_score_within.append(len(feat_train_within))
            with_same_score_across.append(len(feat_train_adversarial_same))
            without_same_score_across.append(len(feat_train_adversarial))

print('-----------------------------')

print('Range for WITHIN-SAMESCORE:\t', min(with_same_score_within), max(with_same_score_within), len(with_same_score_within))
print('Range for WITHIN-DIFFSCORE:\t', min(without_same_score_within), max(without_same_score_within), len(without_same_score_within))

print('Range for CROSS-SAMESCORE:\t', min(with_same_score_across), max(with_same_score_across), len(with_same_score_across))
print('Range for CROSS-DIFFSCORE:\t', min(without_same_score_across), max(without_same_score_across), len(without_same_score_across))

##Range for WITHIN-SAMESCORE:	 2328 2391 122
##Range for WITHIN-DIFFSCORE:	 1497 1838 122
##Range for CROSS-SAMESCORE:	 2331 2388 107
##Range for CROSS-DIFFSCORE:	 1593 2006 107

### Full pairs
##Range for WITHIN-SAMESCORE:	 2250 2391 146
##Range for WITHIN-DIFFSCORE:	 1474 1838 146
##Range for CROSS-SAMESCORE:	 2277 2388 131
##Range for CROSS-DIFFSCORE:	 1655 2178 131