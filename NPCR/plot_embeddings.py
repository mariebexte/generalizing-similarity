import torch
import os
import sys
import umap.umap_ as umap

from tqdm import tqdm
from reader import text_tokenizer
from sklearn.manifold import TSNE
from utils import read_dataframe_asap
from transformers import LongformerTokenizer

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


ASAP_SAVE_MODEL_COMBINATIONS=[(1, 2, 3, 4, 5, 6, 7, 8), (1, 2, 3, 5, 6, 7), (1, 2, 5, 6, 7), (1, 5, 6, 7), (1, 5, 6), (1, 5)]
# ASAP_SAVE_MODEL_COMBINATIONS=[(1, 2, 3, 4, 5, 6, 7, 8), (1, 2, 3, 4, 5, 6, 7), (1, 2, 3, 5, 6, 7), (1, 2, 5, 6, 7), (1, 5, 6, 7), (1, 5, 6), (1, 5)]


def get_embedding(answer_text, model, tokenizer):
    
    answer_text = answer_text.strip()

    sent_tokens_answer = text_tokenizer(answer_text)
    tokenizer_output_answer = tokenizer(sent_tokens_answer, padding='max_length', truncation=True, max_length=1024)

    x0 = torch.LongTensor(np.array([tokenizer_output_answer['input_ids']])).cuda()
    mask_x0 = torch.LongTensor(np.array([tokenizer_output_answer['attention_mask']])).cuda()
    
    return model.embedding(input_ids=x0, attention_mask=mask_x0)[1].detach().cpu().numpy().squeeze()


def plot_embeddings_scatterplot(df_overall, target_path, model_name):

    print(df_overall.columns)
    fig = sns.scatterplot(data=df_overall, x="x", y="y", hue="prompt_id", style='eval_condition', markers={'within': 'o', 'cross': 'X'}, palette='muted')
    sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1))

    # plt.xlim(-40, 40)
    # plt.ylim(-40, 40)

    plt.rcParams['savefig.dpi'] = 500
    plt.tight_layout()
    plt.savefig(os.path.join(target_path, model_name + ".pdf"), transparent=True)

    plt.clf()
    plt.cla()
    plt.close()


def plot_embeddings(model_path, model_name, target_path, combo, test_data_path='../asap'):

    reducer = umap.UMAP()
    dfs = []

    for prompt in range(1,9):

        df = read_dataframe_asap(os.path.join(test_data_path, str(prompt), 'test_scaled.csv'))
        dfs.append(df)


    df_all_test = pd.concat(dfs)
    df_all_test['score_binned'] = df_all_test['score_scaled'].apply(lambda score: int(round(score*5, 0)))
    df_all_test['eval_condition'] = df_all_test['prompt_id'].apply(lambda prompt_id: 'within' if prompt_id in combo else 'cross')

    longformer_file = 'allenai/longformer-base-4096'
    tokenizer = LongformerTokenizer.from_pretrained(longformer_file)
    model = torch.load(os.path.join(model_path, model_name))
    model.cuda()
    model.eval()

    embeddings = []
    for idx, row in tqdm(df_all_test.iterrows(), total=len(df_all_test)):
        answer_text = row['full_text']
        embedding = get_embedding(answer_text, model, tokenizer)
        embeddings.append(embedding)
    
    df_all_test['embeddings'] = embeddings
    embeddings = np.array(embeddings)
    print(embeddings.shape)

    df_all_test['x'], df_all_test['y'] = zip(*reducer.fit_transform(np.array(embeddings)))
    # df_all_test['x'], df_all_test['y'] = zip(*TSNE(n_components=2).fit_transform(np.array(embeddings)))
    plot_embeddings_scatterplot(df_overall=df_all_test, target_path=target_path, model_name='NPCR_longformer')



for combo in ASAP_SAVE_MODEL_COMBINATIONS:

    sample_num = len(combo)
    id_list = [str(element) for element in combo]
    combo_string = '-'.join(id_list)

    for condition in ['standard', 'adversarial']:

        target_path = os.path.join('/results/curve/NPCR_longformer/ASAP_combined/embedding_plots_UMAP', 'sample_' + str(sample_num), combo_string, condition)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        plot_embeddings(model_path=os.path.join('/results/curve/NPCR_longformer/ASAP_combined/sample_' + str(sample_num), combo_string, condition), model_name='best_model.pt', target_path=target_path, combo=combo)

