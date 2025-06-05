import os
import data_prepare

from utils import read_dataframe_asap, ASAP_SCORES
from run_curve import ASAP_INDEPENDENT_SAMPLE, ASAP_INTEGRATED_SAMPLE, get_mixed_data, ASAP_FULL_SAMPLE

# pair trainig data within-prompt
with_same_score_within = []
without_same_score_within = []

# pair training data cross-prompt
with_same_score_across = []
without_same_score_across = []

# single prompts
for prompt in range(1, 9):

    min_score = ASAP_SCORES[prompt][0]
    max_score = ASAP_SCORES[prompt][1]

    df = read_dataframe_asap(os.path.join('../asap', str(prompt), 'train_scaled.csv'))
    df_train, df_val, df_test = data_prepare.prepare_sentence_data_adversarial(df_train=df, df_val=None, df_test=None, max_num=1024)

    # build pairs with same score
    features_train_same_score, _, _ = data_prepare.get_training_pairs(df_train=df_train, training_within_prompt=True, training_with_same_score=True, min_label=min_score, max_label=max_score)

    # build pairs without same score
    features_train, _, _ = data_prepare.get_training_pairs(df_train=df_train, training_within_prompt=True, training_with_same_score=False, min_label=min_score, max_label=max_score)

    with_same_score_within.append(len(features_train_same_score))
    without_same_score_within.append(len(features_train))


# mixed prompts
# for prompt_info in [ASAP_INDEPENDENT_SAMPLE, ASAP_INTEGRATED_SAMPLE]:
for prompt_info in [ASAP_FULL_SAMPLE]:

    for number_of_prompts, prompt_ids in prompt_info.items():

        for prompt_id_element in prompt_ids:

            df_train_mixed = get_mixed_data(prompts=prompt_id_element, target_amount=560, filename='train_scaled.csv', data_path='../asap', persuade=False)
            df_train_mixed, df_val, df_test = data_prepare.prepare_sentence_data_adversarial(df_train=df_train_mixed, df_val=None, df_test=None, max_num=1024)

            # Normal: Training pairs within prompt
            # build pairs with same score
            feat_train_within_same, _, _ = data_prepare.get_training_pairs(df_train=df_train_mixed, training_within_prompt=True, training_with_same_score=True)
            # build pairs without same score
            feat_train_within, _, _ = data_prepare.get_training_pairs(df_train=df_train_mixed, training_within_prompt=True, training_with_same_score=False)

            # Adversarial
            # build pairs with same score
            feat_train_adversarial_same, _, _ = data_prepare.get_training_pairs(df_train=df_train_mixed, training_within_prompt=False, training_with_same_score=True)
            # build pairs without same score
            feat_train_adversarial, _, _ = data_prepare.get_training_pairs(df_train=df_train_mixed, training_within_prompt=False, training_with_same_score=False)

            with_same_score_within.append(len(feat_train_within_same))
            without_same_score_within.append(len(feat_train_within))
            with_same_score_across.append(len(feat_train_adversarial_same))
            without_same_score_across.append(len(feat_train_adversarial))

print('-----------------------------')

print('Range for WITHIN-SAMESCORE:\t', min(with_same_score_within), max(with_same_score_within), len(with_same_score_within))
print('Range for WITHIN-DIFFSCORE:\t', min(without_same_score_within), max(without_same_score_within), len(without_same_score_within))

print('Range for CROSS-SAMESCORE:\t', min(with_same_score_across), max(with_same_score_across), len(with_same_score_across))
print('Range for CROSS-DIFFSCORE:\t', min(without_same_score_across), max(without_same_score_across), len(without_same_score_across))


### SPLIT INTO PROMPT TYPES:
##Range for WITHIN-SAMESCORE:	 1644 1671 30
##Range for WITHIN-DIFFSCORE:	 945 1558 30
##Range for CROSS-SAMESCORE:	 1611 1668 22
##Range for CROSS-DIFFSCORE:	 1181 1644 22

### OVERALL (Combining all prompts:):
##Range for WITHIN-SAMESCORE:	 1608 1671 67
##Range for WITHIN-DIFFSCORE:	 920 1537 67
##Range for CROSS-SAMESCORE:	 1605 1668 59
##Range for CROSS-DIFFSCORE:	 1195 1667 59
