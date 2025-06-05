import os
import sys
import time
import torch
import random
import data_prepare

import numpy as np
import torch.nn as nn
import torch.utils.data as Data

from utils import *
from tqdm import tqdm
from datetime import datetime
from networks.core_networks import npcr_model
from networks.longformer_model import npcr_longformer
from evaluator_core import Evaluator_opti_adversarial, evaluate_finetuned_model


np.random.seed(100)
logger = get_logger("Train...")
        

def train_model(target_path, df_train, df_val, df_test, val_example_size=15, example_size=25, min_label=None, max_label=None, max_num=1024, learning_rate=0.00001, batch_size=6, num_epochs=5, model_name='best_model', longformer=True, training_within_prompt=True, training_with_same_score=False, num_training_pairs=None, col_prompt='prompt_id'):

    start = datetime.now()

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    logging.info('Training: min score is:\t' + str(min_label))
    logging.info('Training: max score is:\t' + str(max_label))

    logging.info('batch size:\t' + str(batch_size))
    logging.info('number of epochs:\t' + str(num_epochs))
    logging.info('learning rate:\t' + str(learning_rate))
    logging.info('number of val examples:\t' + str(val_example_size))
    logging.info('number of test examples:\t' + str(example_size))

    df_train, df_val, df_test = data_prepare.prepare_sentence_data_adversarial(df_train=df_train, df_val=df_val, df_test=df_test, max_num=max_num)

    features_train, masks_train, y_train = data_prepare.get_training_pairs(df_train=df_train, training_within_prompt=training_within_prompt, training_with_same_score=training_with_same_score, min_label=min_label, max_label=max_label, num_training_pairs=num_training_pairs)
    force_cross_prompt_validation_pairs = not training_within_prompt
    features_dev, masks_dev, y_dev_example, y_dev_goal = data_prepare.get_inference_pairs(df=df_val, df_ref=df_train, force_cross_prompt=force_cross_prompt_validation_pairs, example_size=val_example_size, min_label=min_label, max_label=max_label)

    logging.info('number of train examples:\t' + str(len(features_train)))
    logging.info('number of val examples:\t' + str(len(features_dev)))
    print('number of train examples:\t' + str(len(features_train)))
    print('number of val examples:\t' + str(len(features_dev)))

    logger.info("----------------------------------------------------")

    model = npcr_model()
    if longformer:
        model = npcr_longformer()
    model.cuda()

    evl = Evaluator_opti_adversarial(out_dir=target_path, model_name=model_name, features_dev=features_dev, masks_dev=masks_dev,\
        dev_y_example=y_dev_example, dev_y_goal=y_dev_goal, min_label=min_label, max_label=max_label, prompts_array=df_val[col_prompt], example_size=val_example_size)

    logger.info("Train model")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_x0 = [j[0] for j in features_train]
    train_x1 = [j[1] for j in features_train]
    train_x0 = torch.LongTensor(np.array(train_x0))
    train_x1 = torch.LongTensor(np.array(train_x1))

    train_masks_x0 = [m[0] for m in masks_train]
    train_masks_x1 = [m[1] for m in masks_train]
    train_masks_x0 = torch.LongTensor(np.array(train_masks_x0))
    train_masks_x1 = torch.LongTensor(np.array(train_masks_x1))

    torch_dataset = Data.TensorDataset(train_x0, train_x1, train_masks_x0, train_masks_x1, torch.Tensor(y_train))

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    for ii in tqdm(range(num_epochs)):
        logger.info('Epoch %s/%s' % (str(ii+1), num_epochs))
        start_time = time.time()
        for step, (batch_x0, batch_x1, batch_mask_x0, batch_mask_x1, batch_y) in tqdm(enumerate(loader), total=len(loader)):

            optimizer.zero_grad()
            Y_predict = model(batch_x0.cuda(), batch_x1.cuda(), batch_mask_x0.cuda(), batch_mask_x1.cuda())
            loss = loss_fn(Y_predict.squeeze(), batch_y.squeeze().cuda())
            # print('epoch:', ii, 'step:', step, 'loss:', loss.item())
            loss.backward()
            optimizer.step()

        tt_time = time.time() - start_time
        logger.info("Training one epoch in %.3f s" % tt_time)

        model.eval()
        with torch.no_grad():
            evl.evaluate(model, ii, val_example_size, True)
        model.train()

        ttt_time = time.time() - start_time - tt_time
        logger.info("Evaluate one time in %.3f s" % ttt_time)

    logging.info('Full training took:\t' + str(datetime.now() - start))

    if df_test is not None:
        evaluate_finetuned_model(model_path=os.path.join(target_path, model_name), df_ref=df_train, df_test=df_test, min_label=min_label, max_label=max_label, target_path=target_path, max_num=max_num)
        