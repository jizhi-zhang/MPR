import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score, ndcg_score, recall_score
import random 
import os
import copy
import math
import heapq # for retrieval topK
import random
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import time
from collaborative_models import matrixFactorization, sst_pred

from evaluation import evaluate_model_performance_and_naive_fairness_fast_rmse, evaluate_model_performance_and_naive_fairness_fast_partial_valid_rmse

from tqdm import tqdm


def pretrain_epochs_with_resampled_ensemble_sst_reg_eval_unfairness_valid_partial_safe_rmse_thresh_eval(model, df_train, epochs, lr, weight_decay, batch_size, beta, valid_data, test_data, 
                                                                             predicted_sensitive_attr_dict, oracle_sensitive_attr, top_K, fair_reg, 
                                                                             gender_known_male, gender_known_female, device, 
                                                                             evaluation_epoch=3, unsqueeze=False, shuffle=True,random_seed=[0,1,2,3], early_stop=10, rmse_thresh=None):
    # this `safe` version used log sum exp trick to avoid explosion 
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 
    model.train()
    best_val_rmse = 100
    test_rmse_in_that_epoch = 100
    best_epoch = 0
    naive_unfairness_val_in_that_epoch = 100
    naive_unfairness_test_in_that_epoch = 100
    # best_val_unfairness= 1000
    best_val_unfairness=100
    val_rmse_in_that_epoch=100
    achieve_rmse_thresh=0
    stop_epoch = 0
    for i in tqdm(range(epochs)):
        j = 0
        loss_total = 0
        fair_reg_total = 0
        random_id = np.array([i for i in range(len(df_train))])
        if shuffle:
            np.random.shuffle(random_id)
        for batch_i in range(0,np.int64(np.floor(len(df_train)/batch_size))*batch_size,batch_size): # DATA batch
            # data_batch = (df_train[batch_i:(batch_i+batch_size)]).reset_index(drop=True)
            data_batch = df_train.loc[random_id[batch_i:(batch_i+batch_size)]].reset_index(drop=True)
            #train_user_input, train_item_input, train_ratings = get_instances_with_neg_samples(data_batch, probabilities, num_negatives,device)
            # train_user_input, train_item_input, train_ratings = get_instances_with_random_neg_samples(data_batch, num_uniqueLikes, num_negatives,device)
            train_ratings = torch.FloatTensor(np.array(data_batch["label"])).to(device)
            train_user_input = torch.LongTensor(np.array(data_batch["user_id"])).to(device)
            train_item_input = torch.LongTensor(np.array(data_batch["item_id"])).to(device)
            if unsqueeze:
                train_ratings = train_ratings.unsqueeze(1)
            y_hat = model(train_user_input, train_item_input)
            loss = criterion(y_hat, train_ratings.view(-1))
            loss_total += loss.item()
            fair_regulation = torch.Tensor([0.0]).to(device)
            C = torch.Tensor([0.0]).to(device)
            reg_dict = {}
            j = 0
            cnt = 0
            for resample_ratio, resample_seed_dict in  predicted_sensitive_attr_dict.items():
                for seed in resample_seed_dict.keys():
                 # print(seed)
                 resample_df = resample_seed_dict[seed]
                 resampled_user_sst = torch.Tensor(np.array(resample_df.iloc[np.array(data_batch["user_id"])]["gender"])).to(device)
                 resampled_fair_reg = torch.abs((y_hat[ resampled_user_sst == 1]).mean() - (y_hat[ resampled_user_sst == 0]).mean())
                 C = torch.max(C, resampled_fair_reg/beta)
                 reg_dict[cnt] = resampled_fair_reg 
             
                 cnt += 1
                 if False:
                     print(f"Regulation term:[resample_ratio {resample_ratio}] [random seed {random_seed}]: {torch.exp(resampled_fair_reg/beta)}") 

            for resample_ratio, reg in reg_dict.items():
                fair_regulation += torch.exp((reg) /beta - C.detach()) # log sum exp trick
            fair_regulation = fair_reg * beta * (torch.log(fair_regulation) + C.detach())
            fair_reg_total += fair_regulation.item()
            loss = loss + fair_regulation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('epoch: ', i, 'batch: ', j, 'out of: ',np.int64(np.floor(len(df_train)/batch_size)), 'average loss: ',loss.item())
            j = j+1
        print('epoch: ', i, 'average loss: ',loss_total/ j, "fair reg:", fair_reg_total/j)
        if i % evaluation_epoch ==0 :
            rmse_val, naive_unfairness_val = evaluate_model_performance_and_naive_fairness_fast_partial_valid_rmse(model, valid_data, oracle_sensitive_attr, gender_known_male, gender_known_female, top_K, device)
            rmse_test, naive_unfairness_test = evaluate_model_performance_and_naive_fairness_fast_rmse(model, test_data, oracle_sensitive_attr, top_K, device)
            print('epoch: ', i, 'validation RMSE', rmse_val, 'Partial Valid Unfairness:', naive_unfairness_val)
            print('epoch: ', i, 'test RMSE', rmse_test, "Unfairness:", naive_unfairness_test)

            if rmse_val < rmse_thresh:
                achieve_rmse_thresh=1
                if naive_unfairness_val < best_val_unfairness:
                    val_rmse_in_that_epoch = rmse_val
                    test_rmse_in_that_epoch = rmse_test
                    best_model = copy.deepcopy(model)
                    best_epoch = i
                    best_val_unfairness = naive_unfairness_val
                    naive_unfairness_test_in_that_epoch = naive_unfairness_test

            
    if achieve_rmse_thresh == 0:
        # no model achieve rmse thresh
        best_epoch = -1
        best_model=copy.deepcopy(model)


    return val_rmse_in_that_epoch, test_rmse_in_that_epoch, best_val_unfairness, naive_unfairness_test_in_that_epoch, best_epoch, best_model


