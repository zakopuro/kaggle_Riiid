import pandas as pd
import numpy as np
import gc
import os
from tqdm import tqdm
from collections import defaultdict
from base_create_features import Feature, get_arguments, generate_features
from pathlib import Path
import pickle
import dill
import sys
sys.path.append('./')
from utils import data_util,logger
import time
from sklearn.decomposition import PCA

Feature.dir = 'features/all_data'

TARGET = 'answered_correctly'

def _label_encoder(data):
    l_data,_ =data.factorize(sort=True)
    if l_data.max()>32000:
        l_data = l_data.astype('int32')
    else:
        l_data = l_data.astype('int16')

    if data.isnull().sum() > 0:
        l_data = np.where(l_data == -1,np.nan,l_data)
    return l_data


# リークしているけどある程度は仕方ないか。
def target_encoding(data,feature_list):
    group_feature = feature_list.copy()
    group_feature += TARGET
    feature_name = '-'.join(feature_list)
    mean_data = data[group_feature].groupby(feature_list).mean()
    mean_data.columns = [f'{feature_name}_mean']

    return mean_data

class BASE_FIX(Feature):

    def create_features(self):
        self.train = pd.read_feather('./data/train_valid/cv1_train_all.feather')
        self.valid = pd.read_feather('./data/train_valid/cv1_valid_all.feather')

        # self.train = self.train.iloc[-40000000:]
        self.train = self.train.loc[self.train.content_type_id == False].reset_index(drop = True)
        self.valid = self.valid.loc[self.valid.content_type_id == False].reset_index(drop = True)

        # Changing dtype to avoid lightgbm error
        self.train['prior_question_had_explanation'] = self.train.prior_question_had_explanation.fillna(False).astype('int8')
        self.valid['prior_question_had_explanation'] = self.valid.prior_question_had_explanation.fillna(False).astype('int8')

        # Fill prior question elapsed time with the mean
        # prior_question_elapsed_time_mean = self.train['prior_question_elapsed_time'].dropna().mean()
        # self.train['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace = True)
        # self.valid['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace = True)


        qs = pd.read_csv('./data/input/questions.csv')
        tag = qs["tags"].str.split(" ",expand = True)
        tag.columns = ['tags1','tags2','tags3','tags4','tags5','tags6']
        qs = pd.concat([qs,tag],axis=1)
        qs_cmnts = pd.read_csv('./data/input/question_cmnts.csv')
        qs = pd.merge(qs,qs_cmnts,on='question_id',how='left')
        qs = qs.rename(columns={'question_id':'content_id'})

        self.train = pd.merge(self.train,qs,on='content_id',how='left')
        self.valid = pd.merge(self.valid,qs,on='content_id',how='left')

        for i in range(1,7):
            self.train[f'tags{i}'] = self.train[f'tags{i}'].astype(float)
            self.valid[f'tags{i}'] = self.valid[f'tags{i}'].astype(float)

        self.train['part_community'] = self.train['part'] * 10 + self.train['community']
        self.valid['part_community'] = self.valid['part'] * 10 + self.valid['community']

        self.train = self.train.sort_values('row_id').reset_index(drop=True)
        self.valid = self.valid.sort_values('row_id').reset_index(drop=True)



        self.test = pd.DataFrame()

class TAGS(Feature):

    def create_features(self):
        create_feats = ["tags_pca_0", "tags_pca_1",'tags_nan_count','tags_nan_count_mean','tags1_cut_mean','tags1_cut']
        self.train = pd.read_feather(f'./{Feature.dir}/BASE_FIX_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_FIX_valid.feather')

        qs = pd.read_csv('./data/input/questions.csv')
        lst = []
        for tags in qs["tags"]:
            ohe = np.zeros(188)
            if str(tags) != "nan":
                for tag in tags.split():
                    ohe += np.eye(188)[int(tag)]
            lst.append(ohe)
        tags_df = pd.DataFrame(lst, columns=[f"tag_{i}" for i in range(188)]).astype(int)

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(tags_df.values)

        pca_feat_df = pd.DataFrame(X_2d, columns=["tags_pca_0", "tags_pca_1"])
        pca_feat_df["content_id"] = qs["question_id"]

        self.train = pd.merge(self.train,pca_feat_df,on='content_id',how='left')
        self.valid = pd.merge(self.valid,pca_feat_df,on='content_id',how='left')


        # tagsのnanの数
        self.train['tags_nan_count'] = self.train[['tags1','tags2','tags3','tags4','tags5','tags6']].isnull().sum(axis=1)
        self.valid['tags_nan_count'] = self.valid[['tags1','tags2','tags3','tags4','tags5','tags6']].isnull().sum(axis=1)

        self.train.loc[self.train['tags_nan_count'] >= 5, 'tags_nan_count'] = 5
        self.valid.loc[self.valid['tags_nan_count'] >= 5, 'tags_nan_count'] = 5

        tags_nan_count_mean = self.train[[TARGET,'tags_nan_count']].groupby('tags_nan_count').mean().reset_index()
        tags_nan_count_mean.columns = ['tags_nan_count','tags_nan_count_mean']

        self.train = pd.merge(self.train,tags_nan_count_mean,on='tags_nan_count',how='left')
        self.valid = pd.merge(self.valid,tags_nan_count_mean,on='tags_nan_count',how='left')

        # top tags1
        top_tags1 = [143.,  73.,  79.,  96., 131.,   1.,  10.,  80., 133.,   8., 123.,151.,  53.,  23.,   9.,  62., 136.,  74., 157.,  55.]
        self.train['tags1_cut'] = self.train['tags1']
        self.valid['tags1_cut'] = self.valid['tags1']
        self.train.loc[~(self.train['tags1_cut'].isin(top_tags1)) , 'tags1_cut'] = 9999
        self.valid.loc[~(self.valid['tags1_cut'].isin(top_tags1)) , 'tags1_cut'] = 9999
        tags1_cut_mean = self.train[[TARGET,'tags1_cut']].groupby('tags1_cut').mean().reset_index()
        tags1_cut_mean.columns = ['tags1_cut','tags1_cut_mean']
        self.train = pd.merge(self.train,tags1_cut_mean,on='tags1_cut',how='left')
        self.valid = pd.merge(self.valid,tags1_cut_mean,on='tags1_cut',how='left')

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]

        pca_feat_df.to_feather(f'./{Feature.dir}/pca_tags.feather')
        tags_nan_count_mean.to_feather(f'./{Feature.dir}/tags_nan_count_mean.feather')
        tags1_cut_mean.to_feather(f'./{Feature.dir}/tags1_cut_mean.feather')






class GROUP_BY(Feature):

    def create_features(self):
        create_feats = ['ans_part_mean','ans_community_mean','ans_part_community_mean']

        self.train = pd.read_feather(f'./{Feature.dir}/BASE_FIX_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_FIX_valid.feather')

        part_mean = self.train[[TARGET,'part']].groupby('part').mean().reset_index()
        part_mean.columns = ['part','ans_part_mean']
        self.train = pd.merge(self.train,part_mean,on='part',how='left')
        self.valid = pd.merge(self.valid,part_mean,on='part',how='left')
        part_mean.to_feather(f'./{Feature.dir}/part_mean.feather')


        community_mean = self.train[[TARGET,'community']].groupby('community').mean().reset_index()
        community_mean.columns = ['community','ans_community_mean']
        self.train = pd.merge(self.train,community_mean,on='community',how='left')
        self.valid = pd.merge(self.valid,community_mean,on='community',how='left')
        community_mean.to_feather(f'./{Feature.dir}/community_mean.feather')


        part_community_mean = self.train[[TARGET,'part_community']].groupby('part_community').mean().reset_index()
        part_community_mean.columns = ['part_community','ans_part_community_mean']
        self.train = pd.merge(self.train,part_community_mean,on='part_community',how='left')
        self.valid = pd.merge(self.valid,part_community_mean,on='part_community',how='left')
        part_community_mean.to_feather(f'./{Feature.dir}/part_community.feather')

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]















# class LOOP_FIX_TIME6(Feature):

#     def update_tags_time_feats(self,tags_ls,prior_question_elapsed_time,prior_question_had_explanation,
#                               features_dicts):
#         '''
#         0 : count
#         1 : sum
#         2 : elapsed_time_sum
#         3 : explanation_sum
#         '''

#         # if len(features_dicts[feat_list_name][col]) == 0:
#         #     features_dicts[feat_list_name][col] = [0 for _ in range(4)]
#         for tags in tags_ls:
#             tags = int(tags)
#             features_dicts['tags_list'][tags][2] += prior_question_elapsed_time
#             features_dicts['tags_list'][tags][3] += prior_question_had_explanation


#     def update_tags_feats(self,tags_ls,
#                             target,
#                             features_dicts,n):
#         '''
#         0 : count
#         1 : sum
#         2 : elapsed_time_sum
#         3 : explanation_sum
#         '''

#         for tags in tags_ls:
#             tags = int(tags)
#             if len(features_dicts['tags_list'][tags]) == 0:
#                 features_dicts['tags_list'][tags] = [0 for _ in range(n)]
#             features_dicts['tags_list'][tags][1] += target
#             features_dicts['tags_list'][tags][0] += 1


#     def create_tags_feats(self,num,tags_ls,
#                           features_dicts,feats_np_dic):

#         for i in range(6):

#             if i < len(tags_ls):
#                 tags = int(tags_ls[i])
#                 if (len(features_dicts['tags_list'][tags]) >= 4):
#                     if (features_dicts['tags_list'][tags][0] >= 1):
#                         feats_np_dic[f'tags_{i+1}_mean'][num] = features_dicts['tags_list'][tags][1] / features_dicts['tags_list'][tags][0]
#                         feats_np_dic[f'tags_{i+1}_elapsed_time_avg'][num] = features_dicts['tags_list'][tags][2] / features_dicts['tags_list'][tags][0]
#                         feats_np_dic[f'tags_{i+1}_explanation_avg'][num] = features_dicts['tags_list'][tags][3] / features_dicts['tags_list'][tags][0]
#                     else:
#                         feats_np_dic[f'tags_{i+1}_mean'][num] = np.nan
#                         feats_np_dic[f'tags_{i+1}_elapsed_time_avg'][num] = np.nan
#                         feats_np_dic[f'tags_{i+1}_explanation_avg'][num] = np.nan
#                 else:
#                     feats_np_dic[f'tags_{i+1}_mean'][num] = np.nan
#                     feats_np_dic[f'tags_{i+1}_elapsed_time_avg'][num] = np.nan
#                     feats_np_dic[f'tags_{i+1}_explanation_avg'][num] = np.nan

#             else:
#                 feats_np_dic[f'tags_{i+1}_mean'][num] = np.nan
#                 feats_np_dic[f'tags_{i+1}_elapsed_time_avg'][num] = np.nan
#                 feats_np_dic[f'tags_{i+1}_explanation_avg'][num] = np.nan



#     # User 組み合わせ特徴量更新
#     def update_user_arg_feats(self,user_id,col,target,
#                               features_dicts,
#                               ans_user_args_list_name):

#         if len(features_dicts[ans_user_args_list_name][user_id][col]) == 0:
#             features_dicts[ans_user_args_list_name][user_id][col] = [0,0]
#         features_dicts[ans_user_args_list_name][user_id][col][1] += target
#         features_dicts[ans_user_args_list_name][user_id][col][0] += 1


#     # Userと組み合わせ特徴量作成
#     def create_user_args_feats(self,num,user_id,col,
#                                features_dicts,feats_np_dic,
#                                ans_user_args_list_name = None,
#                                ans_user_args_count_name = None,ans_user_args_avg_name = None):

#         if len(features_dicts[ans_user_args_list_name][user_id][col]) == 0:
#             features_dicts[ans_user_args_list_name][user_id][col] = [0,0]

#         if features_dicts[ans_user_args_list_name][user_id][col][0] != 0:
#             feats_np_dic[ans_user_args_avg_name][num] = features_dicts[ans_user_args_list_name][user_id][col][1]/features_dicts[ans_user_args_list_name][user_id][col][0]
#         else:
#             feats_np_dic[ans_user_args_avg_name][num] = np.nan

#         feats_np_dic[ans_user_args_count_name][num] = features_dicts[ans_user_args_list_name][user_id][col][0]



#     # TODO:数値変わるかも
#     def create_first_bundle(self,num,user_id,bundle_id,
#                             features_dicts,
#                             feats_np_dic):

#         if len(features_dicts['user_list'][user_id]) == 0:
#             features_dicts['user_list'][user_id] = [0 for _ in range(4)]
#             features_dicts['user_list'][user_id].append(bundle_id)

#         feats_np_dic['first_bundle'][num] = features_dicts['user_list'][user_id][4]

#     def update_lag_incorrect_feats(self,user_id,timestamp,target,
#                                    features_dicts):

#         if target == 0:
#             if len(features_dicts['lag_user_incorrect_time'][user_id]) == 1:
#                 features_dicts['lag_user_incorrect_time'][user_id].pop(0)
#                 features_dicts['lag_user_incorrect_time'][user_id].append(timestamp)
#             else:
#                 features_dicts['lag_user_incorrect_time'][user_id].append(timestamp)

#     def update_part_lag_incorrect_feats(self,user_id,part,timestamp,target,
#                                         features_dicts):

#         if target == 0:
#             if len(features_dicts['lag_user_part_incorrect_time'][user_id][part]) == 1:
#                 features_dicts['lag_user_part_incorrect_time'][user_id][part].pop(0)
#                 features_dicts['lag_user_part_incorrect_time'][user_id][part].append(timestamp)
#             else:
#                 features_dicts['lag_user_part_incorrect_time'][user_id][part].append(timestamp)


#     def update_args_time_feats(self,col,prior_question_elapsed_time,prior_question_had_explanation,
#                               features_dicts,
#                               feat_list_name):
#         '''
#         0 : count
#         1 : sum
#         2 : elapsed_time_sum
#         3 : explanation_sum
#         '''

#         # if len(features_dicts[feat_list_name][col]) == 0:
#         #     features_dicts[feat_list_name][col] = [0 for _ in range(4)]
#         features_dicts[feat_list_name][col][2] += prior_question_elapsed_time
#         features_dicts[feat_list_name][col][3] += prior_question_had_explanation


#     def update_lag_time_feats(self,user_id,timestamp,
#                              features_dicts):

#         if len(features_dicts['lag_user_time'][user_id]) == 3:
#             features_dicts['lag_user_time'][user_id].pop(0)
#             features_dicts['lag_user_time'][user_id].append(timestamp)
#         else:
#             features_dicts['lag_user_time'][user_id].append(timestamp)


#     def update_create_user_arg_count(self,num,user_id,col,
#                                      features_dicts,feats_np_dic,
#                                      user_args_count_dic_name,
#                                      user_args_count_name):

#         feats_np_dic[user_args_count_name][num] = features_dicts[user_args_count_dic_name][user_id][col]

#         # update
#         features_dicts[user_args_count_dic_name][user_id][col] += 1

#     def update_args_feats(self,col,feat_list_name,
#                             target,
#                             features_dicts,n):
#         '''
#         0 : count
#         1 : sum
#         2 : elapsed_time_sum
#         3 : explanation_sum
#         '''
#         if len(features_dicts[feat_list_name][col]) == 0:
#             features_dicts[feat_list_name][col] = [0 for _ in range(n)]

#         features_dicts[feat_list_name][col][1] += target
#         features_dicts[feat_list_name][col][0] += 1



#     def create_arg_feats(self,num,col,
#                         features_dicts,feats_np_dic,
#                         list_name,
#                         ans_avg_name = None, elapsed_time_avg_name = None, explanation_avg_name = None, elapsed_time_sum_feat_name = None):

#         '''
#         0 : count
#         1 : sum
#         2 : elapsed_time_sum
#         3 : explanation_sum
#         '''

#         if (len(features_dicts[list_name][col]) >= 4):
#             if (features_dicts[list_name][col][0] >= 1):
#                 feats_np_dic[ans_avg_name][num] = features_dicts[list_name][col][1] / features_dicts[list_name][col][0]
#                 feats_np_dic[elapsed_time_avg_name][num] = features_dicts[list_name][col][2] / features_dicts[list_name][col][0]
#                 feats_np_dic[explanation_avg_name][num] = features_dicts[list_name][col][3] / features_dicts[list_name][col][0]
#                 if elapsed_time_sum_feat_name is not None:
#                     feats_np_dic[elapsed_time_sum_feat_name][num] = features_dicts[list_name][col][2]

#             else:
#                 feats_np_dic[ans_avg_name][num] = np.nan
#                 feats_np_dic[elapsed_time_avg_name][num] = np.nan
#                 feats_np_dic[explanation_avg_name][num] = np.nan
#                 if elapsed_time_sum_feat_name is not None:
#                     feats_np_dic[elapsed_time_sum_feat_name][num] = np.nan
#         else:
#             feats_np_dic[ans_avg_name][num] = np.nan
#             feats_np_dic[elapsed_time_avg_name][num] = np.nan
#             feats_np_dic[explanation_avg_name][num] = np.nan
#             if elapsed_time_sum_feat_name is not None:
#                 feats_np_dic[elapsed_time_sum_feat_name][num] = np.nan



#     # ユーザー毎のlag特徴量作成
#     def create_lag_time_feats(self,num,user_id,timestamp,
#                              features_dicts,
#                              feats_np_dic):
#         if len(features_dicts['lag_user_time'][user_id]) == 0:
#             feats_np_dic['lag_time_1'][num] = np.nan
#             feats_np_dic['lag_time_2'][num] = np.nan
#             feats_np_dic['lag_time_3'][num] = np.nan
#         elif len(features_dicts['lag_user_time'][user_id]) == 1:
#             feats_np_dic['lag_time_1'][num] = timestamp - features_dicts['lag_user_time'][user_id][0]
#             feats_np_dic['lag_time_2'][num] = np.nan
#             feats_np_dic['lag_time_3'][num] = np.nan
#         elif len(features_dicts['lag_user_time'][user_id]) == 2:
#             feats_np_dic['lag_time_1'][num] = timestamp - features_dicts['lag_user_time'][user_id][1]
#             feats_np_dic['lag_time_2'][num] = timestamp - features_dicts['lag_user_time'][user_id][0]
#             feats_np_dic['lag_time_3'][num] = np.nan
#         elif len(features_dicts['lag_user_time'][user_id]) == 3:
#             feats_np_dic['lag_time_1'][num] = timestamp - features_dicts['lag_user_time'][user_id][2]
#             feats_np_dic['lag_time_2'][num] = timestamp - features_dicts['lag_user_time'][user_id][1]
#             feats_np_dic['lag_time_3'][num] = timestamp - features_dicts['lag_user_time'][user_id][0]

#         if len(features_dicts['lag_user_incorrect_time'][user_id]) == 0:
#             feats_np_dic['lag_incorrect_time'][num] = np.nan
#         else:
#             feats_np_dic['lag_incorrect_time'][num] = timestamp - features_dicts['lag_user_incorrect_time'][user_id][0]


#     def update_lag_time_mean_feats(self,user_id,timestamp,
#                                    features_dicts):

#         if len(features_dicts['lag_user_time'][user_id]) == 0:
#             features_dicts['lag_time_1_sum'][user_id] = 0

#         elif len(features_dicts['lag_user_time'][user_id]) == 1:
#             features_dicts['lag_time_1_sum'][user_id] += timestamp - features_dicts['lag_user_time'][user_id][0]

#         elif len(features_dicts['lag_user_time'][user_id]) == 2:
#             features_dicts['lag_time_1_sum'][user_id] += timestamp - features_dicts['lag_user_time'][user_id][1]

#         elif len(features_dicts['lag_user_time'][user_id]) == 3:
#             features_dicts['lag_time_1_sum'][user_id] += timestamp - features_dicts['lag_user_time'][user_id][2]


#     def create_lag_time_mean_feats(self,num,user_id,timestamp,
#                                    features_dicts,
#                                    feats_np_dic):

#         if features_dicts['user_list'][user_id][0] >= 2:
#             feats_np_dic['lag_time_1_mean'][num] = features_dicts['lag_time_1_sum'][user_id] / (features_dicts['user_list'][user_id][0] - 1)
#             feats_np_dic['lag_time_1_sum'][num] = features_dicts['lag_time_1_sum'][user_id]
#         else:
#             feats_np_dic['lag_time_1_mean'][num] = np.nan
#             feats_np_dic['lag_time_1_sum'][num] = np.nan



#     def update_part_lag_time_feats(self,user_id,part,timestamp,
#                                   features_dicts):
#         if len(features_dicts['lag_user_part_time'][user_id][part]) == 3:
#             features_dicts['lag_user_part_time'][user_id][part].pop(0)
#             features_dicts['lag_user_part_time'][user_id][part].append(timestamp)
#         else:
#             features_dicts['lag_user_part_time'][user_id][part].append(timestamp)

#     def create_part_lag_time_feats(self,num,user_id,part,timestamp,
#                                    features_dicts,
#                                    feats_np_dic):

#         if len(features_dicts['lag_user_part_time'][user_id][part]) == 0:
#             feats_np_dic['lag_part_time_1'][num] = np.nan
#             feats_np_dic['lag_part_time_2'][num] = np.nan
#             feats_np_dic['lag_part_time_3'][num] = np.nan
#         elif len(features_dicts['lag_user_part_time'][user_id][part]) == 1:
#             feats_np_dic['lag_part_time_1'][num] = timestamp - features_dicts['lag_user_part_time'][user_id][part][0]
#             feats_np_dic['lag_part_time_2'][num] = np.nan
#             feats_np_dic['lag_part_time_3'][num] = np.nan
#         elif len(features_dicts['lag_user_part_time'][user_id][part]) == 2:
#             feats_np_dic['lag_part_time_1'][num] = timestamp - features_dicts['lag_user_part_time'][user_id][part][1]
#             feats_np_dic['lag_part_time_2'][num] = timestamp - features_dicts['lag_user_part_time'][user_id][part][0]
#             feats_np_dic['lag_part_time_3'][num] = np.nan
#         elif len(features_dicts['lag_user_part_time'][user_id][part]) == 3:
#             feats_np_dic['lag_part_time_1'][num] = timestamp - features_dicts['lag_user_part_time'][user_id][part][2]
#             feats_np_dic['lag_part_time_2'][num] = timestamp - features_dicts['lag_user_part_time'][user_id][part][1]
#             feats_np_dic['lag_part_time_3'][num] = timestamp - features_dicts['lag_user_part_time'][user_id][part][0]

# #         if len(features_dicts['lag_user_part_incorrect_time'][user_id][part]) == 0:
# #             feats_np_dic['lag_part_incorrect_time'][num] = np.nan
# #         else:
# #             feats_np_dic['lag_part_incorrect_time'][num] = timestamp - features_dicts['lag_user_part_incorrect_time'][user_id][part][0]




#     def update_feats(self,previous_row,features_dicts,time_update):
#         # メモリ削減のため型変換
#         user_id = int(previous_row[0])
#         target = int(previous_row[1])
#         content_id = int(previous_row[2])
#         prior_question_elapsed_time = previous_row[3]
#         prior_question_had_explanation = int(previous_row[4])
#         timestamp = int(previous_row[5])
#         bundle_id = int(previous_row[6])
#         part = int(previous_row[7])
#         community = int(previous_row[8])
#         tags_ls = previous_row[9]

#         # lag time
#         self.update_lag_incorrect_feats(user_id,timestamp,target,
#                                         features_dicts)

#         # User args feats
#         create_lists = [[part,
#                         'user_part_list',                                                       # dic
#                         'user_part_count','ans_user_part_avg']                                 # np
#                         ]
#         for create_list in create_lists:
#             self.update_user_arg_feats(user_id,create_list[0],target,
#                                                         features_dicts,
#                                                         create_list[1])

#         # arg feats
#         create_lists = [[user_id,
#                         'user_list'],
#                         [content_id,
#                         'content_list']]
#         for create_list in create_lists:
#             self.update_args_feats(create_list[0],create_list[1],
#                                   target,
#                                   features_dicts,
#                                   n=4)

#         self.update_tags_feats(tags_ls,
#                                target,
#                                features_dicts,
#                                n=4)


#         if time_update:
#             self.update_lag_time_mean_feats(user_id,timestamp,
#                                             features_dicts)

#             self.update_lag_time_feats(user_id,timestamp,
#                                     features_dicts)

#             self.update_part_lag_time_feats(user_id,part,timestamp,
#                                             features_dicts)
#             create_lists = [[user_id,
#                             'user_list'],
#                             [content_id,
#                             'content_list']]
#             for create_list in create_lists:
#                 self.update_args_time_feats(create_list[0],prior_question_elapsed_time,prior_question_had_explanation,
#                                 features_dicts,
#                                 create_list[1])

#             self.update_tags_time_feats(tags_ls,prior_question_elapsed_time,prior_question_had_explanation,
#                                         features_dicts)


#         # # rolling mean
#         # self.update_user_ans_list(user_id,target,
#         #                           features_dicts)



#     # 過去分アップデート
#     def update_previous(self,features_dicts,previous_df):
#         for n,previous_row in enumerate(previous_df):
#             # 最後だけupdate
#             if (n+1) == len(previous_df):
#                 self.update_feats(previous_row,features_dicts,time_update=True)
#             else:
#                 self.update_feats(previous_row,features_dicts,time_update=False)


#     # dataframeに格納するnpを一括作成
#     def create_datas(self,df):
#         df_name_float_list = [
#                     # User
#                     'ans_user_avg',
#                     'ans_user_count',
#                     'elapsed_time_user_avg',
#                     'elapsed_time_user_sum',
#                     'explanation_user_avg',
#                     'first_bundle',
#                     # content_id
#                     'ans_content_avg',
#                     'elapsed_time_content_avg',
#                     'explanation_content_avg',
#                     # user lag time
#                     'lag_time_1',
#                     'lag_time_2',
#                     'lag_time_3',
#                     'lag_incorrect_time',
#                     # User Part lag time
#                     'lag_part_time_1',
#                     'lag_part_time_2',
#                     'lag_part_time_3',
#                     # User Part
#                     'ans_user_part_avg',
#                     'lag_time_1_sum',
#                     'lag_time_1_mean'
#         ]

#         tags_mean_name = [f'tags_{i+1}_mean' for i in range(6)]
#         tags_elapsed_time = [f'tags_{i+1}_elapsed_time_avg' for i in range(6)]
#         tags_explanation_name = [f'tags_{i+1}_explanation_avg' for i in range(6)]

#         df_name_float_list = df_name_float_list + tags_mean_name + tags_elapsed_time + tags_explanation_name


#         df_name_int_list = [
#                     # User Content
#                     'user_content_count',
#                     # User Part
#                     'user_part_count'
#         ]

#         feats_np_dic = {}
#         for name in df_name_float_list:
#             feats_np_dic[name] = np.zeros(len(df), dtype = np.float32)
#         for name in df_name_int_list:
#             feats_np_dic[name] = np.zeros(len(df), dtype = np.int32)

#         return feats_np_dic


#     def add_past_feature(self,df, features_dicts,_update = True):
#         # 特徴量格納dicを作成
#         feats_np_dic = self.create_datas(df)
#         previous_bundle_id = None
#         previous_user_id = None
#         previous_row = None
#         update_cnt = 0
#         previous_df = []


#         for num, row in enumerate(tqdm(df[['user_id', 'answered_correctly', 'content_id', 'prior_question_elapsed_time',
#                                             'prior_question_had_explanation', 'timestamp','bundle_id','part','community','tags_ls']].values)):
#             # メモリ削減のため型変換
#             user_id = int(row[0])
#             target = int(row[1])
#             content_id = int(row[2])
#             prior_question_elapsed_time = row[3]
#             prior_question_had_explanation = int(row[4])
#             timestamp = int(row[5])
#             bundle_id = int(row[6])
#             part = int(row[7])
#             community = int(row[8])
#             tags_ls = row[9]

#             update = _update
#             # 前回とbundle_idが同じ時は更新しない
#             if (previous_bundle_id == bundle_id) & (previous_user_id == user_id) & (_update):
#                 update = False
#                 if update_cnt == 0:
#                     previous_df.append(previous_row)
#                 previous_df.append(row)
#                 update_cnt += 1

#             # 溜まっていたら過去情報をupdate
#             if (update) & (len(previous_df) > 0):
#                 self.update_previous(features_dicts,previous_df)
#                 previous_df = []
#                 update_cnt = 0
#                 update = False

#             if (update) & (previous_row is not None):
#                 self.update_feats(previous_row,features_dicts,time_update=True)

#             previous_bundle_id = bundle_id
#             previous_user_id = user_id
#             previous_row = row


#             # Args
#             create_lists = [[user_id,
#                             'user_list',                 # dic
#                             'ans_user_avg','elapsed_time_user_avg','explanation_user_avg','elapsed_time_user_sum'],                                 # np
#                             [content_id,
#                             'content_list',     # dic
#                             'ans_content_avg','elapsed_time_content_avg','explanation_content_avg',None]                        # np
#                             ]


#             for create_list in create_lists:
#                 self.create_arg_feats(num,create_list[0],
#                                     features_dicts,feats_np_dic,
#                                     create_list[1],
#                                     create_list[2],create_list[3],create_list[4],create_list[5])

#                 # # 常に更新
#                 # self.update_args_time_feats(create_list[0],prior_question_elapsed_time,prior_question_had_explanation,
#                 #                             features_dicts,
#                 #                             create_list[1])

#             self.create_tags_feats(num,tags_ls,
#                                    features_dicts,feats_np_dic)


#             # First bundle
#             self.create_first_bundle(num,user_id,bundle_id,
#                                     features_dicts,
#                                     feats_np_dic)


#             # lag time
#             self.create_lag_time_feats(num,user_id,timestamp,
#                                     features_dicts,
#                                     feats_np_dic)

#             self.create_part_lag_time_feats(num,user_id,part,timestamp,
#                                             features_dicts,
#                                             feats_np_dic)

#             # # rolling mean
#             # self.create_user_ans_rolling_mean(num,user_id,target,
#             #                                   features_dicts,
#             #                                   feats_np_dic)


#             # User args feats
#             # ------------------------------------------------------------------
#             create_lists = [[part,
#                             'user_part_list',                                                       # dic
#                             'user_part_count','ans_user_part_avg'],                                 # np
#                             ]
#             for create_list in create_lists:
#                 self.create_user_args_feats(num,user_id,create_list[0],
#                                             features_dicts,feats_np_dic,
#                                             create_list[1],
#                                             create_list[2],create_list[3]
#                                             )



#             # TODO:Updateに回すべきか
#             # User args count
#             # ------------------------------------------------------------------
#             create_lists = [[content_id,
#                             'user_content_count',                            # dic
#                             'user_content_count']                            # np
#                             ]
#             for create_list in create_lists:
#                 self.update_create_user_arg_count(num,user_id,create_list[0],
#                                                   features_dicts,feats_np_dic,
#                                                   create_list[1],
#                                                   create_list[2])

#             # TODO : カウントは毎回更新するべきか
#             # User count
#             feats_np_dic['ans_user_count'][num] = features_dicts['user_list'][user_id][0]


#             self.create_lag_time_mean_feats(num,user_id,timestamp,
#                                             features_dicts,
#                                             feats_np_dic)

#             # Update
#             # ------------------------------------------------------------------
#             # lag time
#             # ------------------------------------------------------------------
#             # self.update_lag_time_feats(user_id,timestamp,
#             #                         features_dicts)

#             # self.update_part_lag_time_feats(user_id,part,timestamp,
#             #                                 features_dicts)

#             # count
#             # ------------------------------------------------------------------
#             # features_dicts['ans_user_count'][user_id] += 1
#             # features_dicts['ans_content_count'][content_id] += 1



#         loop_feats_df = pd.DataFrame(feats_np_dic)

#         tags_mean_name = [f'tags_{i+1}_mean' for i in range(6)]
#         tags_elapsed_time = [f'tags_{i+1}_elapsed_time_avg' for i in range(6)]
#         tags_explanation_name = [f'tags_{i+1}_explanation_avg' for i in range(6)]

#         loop_feats_df['tags_mean'] = loop_feats_df[tags_mean_name].mean(axis=1)
#         loop_feats_df['tags_elapsed_time_avg'] = loop_feats_df[tags_elapsed_time].mean(axis=1)
#         loop_feats_df['tags_explanation_avg'] = loop_feats_df[tags_explanation_name].mean(axis=1)

#         df = pd.concat([df, loop_feats_df], axis = 1)
#         create_feats = list(feats_np_dic.keys()) + ['tags_mean','tags_elapsed_time_avg','tags_explanation_avg']
#         return df,create_feats


#     def create_dics(self):
#         features_dicts = {}
#         list_name = [
#                     # User
#                     'user_list', # 'ans_user_count','ans_user_sum','elapsed_time_user_sum','explanation_user_sum','user_first_bundle'
#                     # content_id
#                     'content_list', # 'ans_content_count','ans_content_sum','elapsed_time_content_sum','explanation_content_sum'
#                     # User Time
#                     'lag_user_time',
#                     'lag_user_incorrect_time',

#                     'tags_list'
#         ]

#         int_name = [
#                     'lag_time_1_sum'
#         ]


#         lambda_int_name = [
#                     # User content_id
#                     'user_content_count',
#                     # User bundle_id
#                     'user_bundle_count'
#         ]

#         lambda_list_name = [
#                     # User Part Time
#                     'lag_user_part_time',
#                     'lag_user_part_incorrect_time',
#                     # User Part
#                     'user_part_list'
#         ]


#         for name in list_name:
#             features_dicts[name] = defaultdict(list)

#         for name in int_name:
#             features_dicts[name] = defaultdict(int)

#         for name in lambda_int_name:
#             features_dicts[name] = defaultdict(lambda: defaultdict(int))

#         for name in lambda_list_name:
#             features_dicts[name] = defaultdict(lambda: defaultdict(list))


#         return features_dicts


#     def create_features(self):

#         self.train = pd.read_feather(f'./{Feature.dir}/BASE_FIX_train.feather')
#         self.valid = pd.read_feather(f'./{Feature.dir}/BASE_FIX_valid.feather')
#         self.train['prior_question_elapsed_time'] = self.train['prior_question_elapsed_time'].fillna(0)
#         self.valid['prior_question_elapsed_time'] = self.valid['prior_question_elapsed_time'].fillna(0)



#         questions = pd.read_csv('./data/input/questions.csv')
#         inv_rows = questions[questions['tags'].apply(type) == float].index
#         questions.at[inv_rows, 'tags'] = ''
#         # split tag string into list of ints
#         questions['tags_ls'] = questions['tags'].apply(lambda x: np.array(x.split()).astype(int))
#         questions = questions.rename(columns={'question_id':'content_id'})
#         self.train = pd.merge(self.train,questions[['content_id','tags_ls']],on='content_id',how='left')
#         self.valid = pd.merge(self.valid,questions[['content_id','tags_ls']],on='content_id',how='left')



#         features_dicts = self.create_dics()

#         self.train , _ = self.add_past_feature(self.train, features_dicts)
#         self.valid , create_feats = self.add_past_feature(self.valid, features_dicts)


#         self.train = self.train[create_feats]
#         self.valid = self.valid[create_feats]


#         with open(f'./features/all_data/loop_feats10.dill','wb') as f:
#             dill.dump(features_dicts,f)








class TAGS_MEAN(Feature):

    def update_tags_time_feats(self,tags_ls,prior_question_elapsed_time,prior_question_had_explanation,
                              features_dicts):
        '''
        0 : count
        1 : sum
        2 : elapsed_time_sum
        3 : explanation_sum
        '''

        # if len(features_dicts[feat_list_name][col]) == 0:
        #     features_dicts[feat_list_name][col] = [0 for _ in range(4)]
        for tags in tags_ls:
            tags = int(tags)
            features_dicts['tags_list'][tags][2] += prior_question_elapsed_time
            features_dicts['tags_list'][tags][3] += prior_question_had_explanation


    def update_tags_feats(self,tags_ls,
                            target,
                            features_dicts,n):
        '''
        0 : count
        1 : sum
        2 : elapsed_time_sum
        3 : explanation_sum
        '''

        for tags in tags_ls:
            tags = int(tags)
            if len(features_dicts['tags_list'][tags]) == 0:
                features_dicts['tags_list'][tags] = [0 for _ in range(n)]
            features_dicts['tags_list'][tags][1] += target
            features_dicts['tags_list'][tags][0] += 1


    def create_tags_feats(self,num,tags_ls,
                          features_dicts,feats_np_dic):

        for i in range(6):

            if i < len(tags_ls):
                tags = int(tags_ls[i])
                if (len(features_dicts['tags_list'][tags]) >= 4):
                    if (features_dicts['tags_list'][tags][0] >= 1):
                        feats_np_dic[f'tags_{i+1}_mean'][num] = features_dicts['tags_list'][tags][1] / features_dicts['tags_list'][tags][0]
                        feats_np_dic[f'tags_{i+1}_elapsed_time_avg'][num] = features_dicts['tags_list'][tags][2] / features_dicts['tags_list'][tags][0]
                        feats_np_dic[f'tags_{i+1}_explanation_avg'][num] = features_dicts['tags_list'][tags][3] / features_dicts['tags_list'][tags][0]
                    else:
                        feats_np_dic[f'tags_{i+1}_mean'][num] = np.nan
                        feats_np_dic[f'tags_{i+1}_elapsed_time_avg'][num] = np.nan
                        feats_np_dic[f'tags_{i+1}_explanation_avg'][num] = np.nan
                else:
                    feats_np_dic[f'tags_{i+1}_mean'][num] = np.nan
                    feats_np_dic[f'tags_{i+1}_elapsed_time_avg'][num] = np.nan
                    feats_np_dic[f'tags_{i+1}_explanation_avg'][num] = np.nan

            else:
                feats_np_dic[f'tags_{i+1}_mean'][num] = np.nan
                feats_np_dic[f'tags_{i+1}_elapsed_time_avg'][num] = np.nan
                feats_np_dic[f'tags_{i+1}_explanation_avg'][num] = np.nan



    # User 組み合わせ特徴量更新
    def update_user_arg_feats(self,user_id,col,target,
                              features_dicts,
                              ans_user_args_list_name):

        if len(features_dicts[ans_user_args_list_name][user_id][col]) == 0:
            features_dicts[ans_user_args_list_name][user_id][col] = [0,0]
        features_dicts[ans_user_args_list_name][user_id][col][1] += target
        features_dicts[ans_user_args_list_name][user_id][col][0] += 1


    # Userと組み合わせ特徴量作成
    def create_user_args_feats(self,num,user_id,col,
                               features_dicts,feats_np_dic,
                               ans_user_args_list_name = None,
                               ans_user_args_count_name = None,ans_user_args_avg_name = None):

        if len(features_dicts[ans_user_args_list_name][user_id][col]) == 0:
            features_dicts[ans_user_args_list_name][user_id][col] = [0,0]

        if features_dicts[ans_user_args_list_name][user_id][col][0] != 0:
            feats_np_dic[ans_user_args_avg_name][num] = features_dicts[ans_user_args_list_name][user_id][col][1]/features_dicts[ans_user_args_list_name][user_id][col][0]
        else:
            feats_np_dic[ans_user_args_avg_name][num] = np.nan

        feats_np_dic[ans_user_args_count_name][num] = features_dicts[ans_user_args_list_name][user_id][col][0]



    # TODO:数値変わるかも
    def create_first_bundle(self,num,user_id,bundle_id,
                            features_dicts,
                            feats_np_dic):

        if len(features_dicts['user_list'][user_id]) == 0:
            features_dicts['user_list'][user_id] = [0 for _ in range(4)]
            features_dicts['user_list'][user_id].append(bundle_id)

        feats_np_dic['first_bundle'][num] = features_dicts['user_list'][user_id][4]

    def update_lag_incorrect_feats(self,user_id,timestamp,target,
                                   features_dicts):

        if target == 0:
            if len(features_dicts['lag_user_incorrect_time'][user_id]) == 1:
                features_dicts['lag_user_incorrect_time'][user_id].pop(0)
                features_dicts['lag_user_incorrect_time'][user_id].append(timestamp)
            else:
                features_dicts['lag_user_incorrect_time'][user_id].append(timestamp)

    def update_part_lag_incorrect_feats(self,user_id,part,timestamp,target,
                                        features_dicts):

        if target == 0:
            if len(features_dicts['lag_user_part_incorrect_time'][user_id][part]) == 1:
                features_dicts['lag_user_part_incorrect_time'][user_id][part].pop(0)
                features_dicts['lag_user_part_incorrect_time'][user_id][part].append(timestamp)
            else:
                features_dicts['lag_user_part_incorrect_time'][user_id][part].append(timestamp)


    def update_args_time_feats(self,col,prior_question_elapsed_time,prior_question_had_explanation,
                              features_dicts,
                              feat_list_name):
        '''
        0 : count
        1 : sum
        2 : elapsed_time_sum
        3 : explanation_sum
        '''

        # if len(features_dicts[feat_list_name][col]) == 0:
        #     features_dicts[feat_list_name][col] = [0 for _ in range(4)]
        features_dicts[feat_list_name][col][2] += prior_question_elapsed_time
        features_dicts[feat_list_name][col][3] += prior_question_had_explanation


    def update_lag_time_feats(self,user_id,timestamp,
                             features_dicts):

        if len(features_dicts['lag_user_time'][user_id]) == 3:
            features_dicts['lag_user_time'][user_id].pop(0)
            features_dicts['lag_user_time'][user_id].append(timestamp)
        else:
            features_dicts['lag_user_time'][user_id].append(timestamp)


    def update_create_user_arg_count(self,num,user_id,col,
                                     features_dicts,feats_np_dic,
                                     user_args_count_dic_name,
                                     user_args_count_name):

        feats_np_dic[user_args_count_name][num] = features_dicts[user_args_count_dic_name][user_id][col]

        # update
        features_dicts[user_args_count_dic_name][user_id][col] += 1

    def update_args_feats(self,col,feat_list_name,
                            target,
                            features_dicts,n):
        '''
        0 : count
        1 : sum
        2 : elapsed_time_sum
        3 : explanation_sum
        '''
        if len(features_dicts[feat_list_name][col]) == 0:
            features_dicts[feat_list_name][col] = [0 for _ in range(n)]

        features_dicts[feat_list_name][col][1] += target
        features_dicts[feat_list_name][col][0] += 1



    def create_arg_feats(self,num,col,
                        features_dicts,feats_np_dic,
                        list_name,
                        ans_avg_name = None, elapsed_time_avg_name = None, explanation_avg_name = None, elapsed_time_sum_feat_name = None):

        '''
        0 : count
        1 : sum
        2 : elapsed_time_sum
        3 : explanation_sum
        '''

        if (len(features_dicts[list_name][col]) >= 4):
            if (features_dicts[list_name][col][0] >= 1):
                feats_np_dic[ans_avg_name][num] = features_dicts[list_name][col][1] / features_dicts[list_name][col][0]
                feats_np_dic[elapsed_time_avg_name][num] = features_dicts[list_name][col][2] / features_dicts[list_name][col][0]
                feats_np_dic[explanation_avg_name][num] = features_dicts[list_name][col][3] / features_dicts[list_name][col][0]
                if elapsed_time_sum_feat_name is not None:
                    feats_np_dic[elapsed_time_sum_feat_name][num] = features_dicts[list_name][col][2]

            else:
                feats_np_dic[ans_avg_name][num] = np.nan
                feats_np_dic[elapsed_time_avg_name][num] = np.nan
                feats_np_dic[explanation_avg_name][num] = np.nan
                if elapsed_time_sum_feat_name is not None:
                    feats_np_dic[elapsed_time_sum_feat_name][num] = np.nan
        else:
            feats_np_dic[ans_avg_name][num] = np.nan
            feats_np_dic[elapsed_time_avg_name][num] = np.nan
            feats_np_dic[explanation_avg_name][num] = np.nan
            if elapsed_time_sum_feat_name is not None:
                feats_np_dic[elapsed_time_sum_feat_name][num] = np.nan



    # ユーザー毎のlag特徴量作成
    def create_lag_time_feats(self,num,user_id,timestamp,
                             features_dicts,
                             feats_np_dic):
        if len(features_dicts['lag_user_time'][user_id]) == 0:
            feats_np_dic['lag_time_1'][num] = np.nan
            feats_np_dic['lag_time_2'][num] = np.nan
            feats_np_dic['lag_time_3'][num] = np.nan
        elif len(features_dicts['lag_user_time'][user_id]) == 1:
            feats_np_dic['lag_time_1'][num] = timestamp - features_dicts['lag_user_time'][user_id][0]
            feats_np_dic['lag_time_2'][num] = np.nan
            feats_np_dic['lag_time_3'][num] = np.nan
        elif len(features_dicts['lag_user_time'][user_id]) == 2:
            feats_np_dic['lag_time_1'][num] = timestamp - features_dicts['lag_user_time'][user_id][1]
            feats_np_dic['lag_time_2'][num] = timestamp - features_dicts['lag_user_time'][user_id][0]
            feats_np_dic['lag_time_3'][num] = np.nan
        elif len(features_dicts['lag_user_time'][user_id]) == 3:
            feats_np_dic['lag_time_1'][num] = timestamp - features_dicts['lag_user_time'][user_id][2]
            feats_np_dic['lag_time_2'][num] = timestamp - features_dicts['lag_user_time'][user_id][1]
            feats_np_dic['lag_time_3'][num] = timestamp - features_dicts['lag_user_time'][user_id][0]

        if len(features_dicts['lag_user_incorrect_time'][user_id]) == 0:
            feats_np_dic['lag_incorrect_time'][num] = np.nan
        else:
            feats_np_dic['lag_incorrect_time'][num] = timestamp - features_dicts['lag_user_incorrect_time'][user_id][0]


    def update_lag_time_mean_feats(self,user_id,timestamp,
                                   features_dicts):

        if len(features_dicts['lag_user_time'][user_id]) == 0:
            features_dicts['lag_time_1_sum'][user_id] = 0

        elif len(features_dicts['lag_user_time'][user_id]) == 1:
            features_dicts['lag_time_1_sum'][user_id] += timestamp - features_dicts['lag_user_time'][user_id][0]

        elif len(features_dicts['lag_user_time'][user_id]) == 2:
            features_dicts['lag_time_1_sum'][user_id] += timestamp - features_dicts['lag_user_time'][user_id][1]

        elif len(features_dicts['lag_user_time'][user_id]) == 3:
            features_dicts['lag_time_1_sum'][user_id] += timestamp - features_dicts['lag_user_time'][user_id][2]


    def create_lag_time_mean_feats(self,num,user_id,timestamp,
                                   features_dicts,
                                   feats_np_dic):

        if features_dicts['user_list'][user_id][0] >= 2:
            feats_np_dic['lag_time_1_mean'][num] = features_dicts['lag_time_1_sum'][user_id] / (features_dicts['user_list'][user_id][0] - 1)
            feats_np_dic['lag_time_1_sum'][num] = features_dicts['lag_time_1_sum'][user_id]
        else:
            feats_np_dic['lag_time_1_mean'][num] = np.nan
            feats_np_dic['lag_time_1_sum'][num] = np.nan



    def update_part_lag_time_feats(self,user_id,part,timestamp,
                                  features_dicts):
        if len(features_dicts['lag_user_part_time'][user_id][part]) == 3:
            features_dicts['lag_user_part_time'][user_id][part].pop(0)
            features_dicts['lag_user_part_time'][user_id][part].append(timestamp)
        else:
            features_dicts['lag_user_part_time'][user_id][part].append(timestamp)

    def create_part_lag_time_feats(self,num,user_id,part,timestamp,
                                   features_dicts,
                                   feats_np_dic):

        if len(features_dicts['lag_user_part_time'][user_id][part]) == 0:
            feats_np_dic['lag_part_time_1'][num] = np.nan
            feats_np_dic['lag_part_time_2'][num] = np.nan
            feats_np_dic['lag_part_time_3'][num] = np.nan
        elif len(features_dicts['lag_user_part_time'][user_id][part]) == 1:
            feats_np_dic['lag_part_time_1'][num] = timestamp - features_dicts['lag_user_part_time'][user_id][part][0]
            feats_np_dic['lag_part_time_2'][num] = np.nan
            feats_np_dic['lag_part_time_3'][num] = np.nan
        elif len(features_dicts['lag_user_part_time'][user_id][part]) == 2:
            feats_np_dic['lag_part_time_1'][num] = timestamp - features_dicts['lag_user_part_time'][user_id][part][1]
            feats_np_dic['lag_part_time_2'][num] = timestamp - features_dicts['lag_user_part_time'][user_id][part][0]
            feats_np_dic['lag_part_time_3'][num] = np.nan
        elif len(features_dicts['lag_user_part_time'][user_id][part]) == 3:
            feats_np_dic['lag_part_time_1'][num] = timestamp - features_dicts['lag_user_part_time'][user_id][part][2]
            feats_np_dic['lag_part_time_2'][num] = timestamp - features_dicts['lag_user_part_time'][user_id][part][1]
            feats_np_dic['lag_part_time_3'][num] = timestamp - features_dicts['lag_user_part_time'][user_id][part][0]

#         if len(features_dicts['lag_user_part_incorrect_time'][user_id][part]) == 0:
#             feats_np_dic['lag_part_incorrect_time'][num] = np.nan
#         else:
#             feats_np_dic['lag_part_incorrect_time'][num] = timestamp - features_dicts['lag_user_part_incorrect_time'][user_id][part][0]




    def update_feats(self,previous_row,features_dicts,time_update):
        # メモリ削減のため型変換
        user_id = int(previous_row[0])
        target = int(previous_row[1])
        content_id = int(previous_row[2])
        prior_question_elapsed_time = previous_row[3]
        prior_question_had_explanation = int(previous_row[4])
        timestamp = int(previous_row[5])
        bundle_id = int(previous_row[6])
        part = int(previous_row[7])
        community = int(previous_row[8])
        tags_ls = previous_row[9]


        self.update_tags_feats(tags_ls,
                               target,
                               features_dicts,
                               n=4)

        if time_update:
            self.update_tags_time_feats(tags_ls,prior_question_elapsed_time,prior_question_had_explanation,
                                        features_dicts)


        # # rolling mean
        # self.update_user_ans_list(user_id,target,
        #                           features_dicts)



    # 過去分アップデート
    def update_previous(self,features_dicts,previous_df):
        for n,previous_row in enumerate(previous_df):
            # 最後だけupdate
            if (n+1) == len(previous_df):
                self.update_feats(previous_row,features_dicts,time_update=True)
            else:
                self.update_feats(previous_row,features_dicts,time_update=False)


    # dataframeに格納するnpを一括作成
    def create_datas(self,df):
        df_name_float_list = [
                    # User
                    'ans_user_avg',
                    'ans_user_count',
                    'elapsed_time_user_avg',
                    'elapsed_time_user_sum',
                    'explanation_user_avg',
                    'first_bundle',
                    # content_id
                    'ans_content_avg',
                    'elapsed_time_content_avg',
                    'explanation_content_avg',
                    # user lag time
                    'lag_time_1',
                    'lag_time_2',
                    'lag_time_3',
                    'lag_incorrect_time',
                    # User Part lag time
                    'lag_part_time_1',
                    'lag_part_time_2',
                    'lag_part_time_3',
                    # User Part
                    'ans_user_part_avg',
                    'lag_time_1_sum',
                    'lag_time_1_mean'
        ]

        tags_mean_name = [f'tags_{i+1}_mean' for i in range(6)]
        tags_elapsed_time = [f'tags_{i+1}_elapsed_time_avg' for i in range(6)]
        tags_explanation_name = [f'tags_{i+1}_explanation_avg' for i in range(6)]

        df_name_float_list = df_name_float_list + tags_mean_name + tags_elapsed_time + tags_explanation_name


        df_name_int_list = [
                    # User Content
                    'user_content_count',
                    # User Part
                    'user_part_count'
        ]

        feats_np_dic = {}
        for name in df_name_float_list:
            feats_np_dic[name] = np.zeros(len(df), dtype = np.float32)
        for name in df_name_int_list:
            feats_np_dic[name] = np.zeros(len(df), dtype = np.int32)

        return feats_np_dic


    def add_past_feature(self,df, features_dicts,_update = True):
        # 特徴量格納dicを作成
        feats_np_dic = self.create_datas(df)
        previous_bundle_id = None
        previous_user_id = None
        previous_row = None
        update_cnt = 0
        previous_df = []


        for num, row in enumerate(tqdm(df[['user_id', 'answered_correctly', 'content_id', 'prior_question_elapsed_time',
                                            'prior_question_had_explanation', 'timestamp','bundle_id','part','community','tags_ls']].values)):
            # メモリ削減のため型変換
            user_id = int(row[0])
            target = int(row[1])
            content_id = int(row[2])
            prior_question_elapsed_time = row[3]
            prior_question_had_explanation = int(row[4])
            timestamp = int(row[5])
            bundle_id = int(row[6])
            part = int(row[7])
            community = int(row[8])
            tags_ls = row[9]

            update = _update
            # 前回とbundle_idが同じ時は更新しない
            if (previous_bundle_id == bundle_id) & (previous_user_id == user_id) & (_update):
                update = False
                if update_cnt == 0:
                    previous_df.append(previous_row)
                previous_df.append(row)
                update_cnt += 1

            # 溜まっていたら過去情報をupdate
            if (update) & (len(previous_df) > 0):
                self.update_previous(features_dicts,previous_df)
                previous_df = []
                update_cnt = 0
                update = False

            if (update) & (previous_row is not None):
                self.update_feats(previous_row,features_dicts,time_update=True)

            previous_bundle_id = bundle_id
            previous_user_id = user_id
            previous_row = row


            self.create_tags_feats(num,tags_ls,
                                   features_dicts,feats_np_dic)


        loop_feats_df = pd.DataFrame(feats_np_dic)

        tags_mean_name = [f'tags_{i+1}_mean' for i in range(6)]
        tags_elapsed_time = [f'tags_{i+1}_elapsed_time_avg' for i in range(6)]
        tags_explanation_name = [f'tags_{i+1}_explanation_avg' for i in range(6)]

        loop_feats_df['tags_mean'] = loop_feats_df[tags_mean_name].mean(axis=1)
        loop_feats_df['tags_elapsed_time_avg'] = loop_feats_df[tags_elapsed_time].mean(axis=1)
        loop_feats_df['tags_explanation_avg'] = loop_feats_df[tags_explanation_name].mean(axis=1)

        df = pd.concat([df, loop_feats_df], axis = 1)
        create_feats = list(feats_np_dic.keys()) + ['tags_mean','tags_elapsed_time_avg','tags_explanation_avg']
        return df,create_feats


    def create_dics(self):
        features_dicts = {}
        list_name = [
                    # User
                    'user_list', # 'ans_user_count','ans_user_sum','elapsed_time_user_sum','explanation_user_sum','user_first_bundle'
                    # content_id
                    'content_list', # 'ans_content_count','ans_content_sum','elapsed_time_content_sum','explanation_content_sum'
                    # User Time
                    'lag_user_time',
                    'lag_user_incorrect_time',

                    'tags_list'
        ]

        int_name = [
                    'lag_time_1_sum'
        ]


        lambda_int_name = [
                    # User content_id
                    'user_content_count',
                    # User bundle_id
                    'user_bundle_count'
        ]

        lambda_list_name = [
                    # User Part Time
                    'lag_user_part_time',
                    'lag_user_part_incorrect_time',
                    # User Part
                    'user_part_list'
        ]


        for name in list_name:
            features_dicts[name] = defaultdict(list)

        for name in int_name:
            features_dicts[name] = defaultdict(int)

        for name in lambda_int_name:
            features_dicts[name] = defaultdict(lambda: defaultdict(int))

        for name in lambda_list_name:
            features_dicts[name] = defaultdict(lambda: defaultdict(list))


        return features_dicts


    def create_features(self):

        self.train = pd.read_feather(f'./{Feature.dir}/BASE_FIX_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_FIX_valid.feather')
        self.train['prior_question_elapsed_time'] = self.train['prior_question_elapsed_time'].fillna(0)
        self.valid['prior_question_elapsed_time'] = self.valid['prior_question_elapsed_time'].fillna(0)



        questions = pd.read_csv('./data/input/questions.csv')
        inv_rows = questions[questions['tags'].apply(type) == float].index
        questions.at[inv_rows, 'tags'] = ''
        # split tag string into list of ints
        questions['tags_ls'] = questions['tags'].apply(lambda x: np.array(x.split()).astype(int))
        questions = questions.rename(columns={'question_id':'content_id'})
        self.train = pd.merge(self.train,questions[['content_id','tags_ls']],on='content_id',how='left')
        self.valid = pd.merge(self.valid,questions[['content_id','tags_ls']],on='content_id',how='left')



        features_dicts = self.create_dics()

        self.train , _ = self.add_past_feature(self.train, features_dicts)
        self.valid , create_feats = self.add_past_feature(self.valid, features_dicts)


        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]


        with open(f'./features/all_data/tags_loop_feats.dill','wb') as f:
            dill.dump(features_dicts,f)















class BUNDLE_ID(Feature):

    def create_features(self):
        create_feats = ['first_bundle_cut','first_bundle_cut_mean']

        self.train = pd.read_feather(f'./{Feature.dir}/BASE_FIX_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_FIX_valid.feather')

        train_loop = pd.read_feather(f'./{Feature.dir}/LOOP_FIX2_train.feather')[['first_bundle']]
        valid_loop = pd.read_feather(f'./{Feature.dir}/LOOP_FIX2_valid.feather')[['first_bundle']]

        self.train = pd.concat([self.train,train_loop],axis=1)
        self.valid = pd.concat([self.valid,valid_loop],axis=1)

        top_first_bundle = [7900,128,5692,7876,2063,3363,1278,175,1232,4528]
        self.train['first_bundle_cut'] = self.train['first_bundle']
        self.valid['first_bundle_cut'] = self.valid['first_bundle']
        self.train.loc[~(self.train['first_bundle_cut'].isin(top_first_bundle)) ,'first_bundle_cut'] = 9999
        self.valid.loc[~(self.valid['first_bundle_cut'].isin(top_first_bundle)) ,'first_bundle_cut'] = 9999

        first_bundle_cut_mean = self.train[[TARGET,'first_bundle_cut']].groupby('first_bundle_cut').mean().reset_index()
        first_bundle_cut_mean.columns = ['first_bundle_cut','first_bundle_cut_mean']
        first_bundle_cut_mean.to_feather(f'./{Feature.dir}/first_bundle_cut_mean.feather')


        self.train = pd.merge(self.train,first_bundle_cut_mean,on='first_bundle_cut',how='left')
        self.valid = pd.merge(self.valid,first_bundle_cut_mean,on='first_bundle_cut',how='left')

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]

class ROLLING_MEAN2(Feature):

    # rolling mean
    def create_user_ans_rolling_mean(self,num,user_id,timestamp,
                                    features_dicts,
                                    feats_np_dic,
                                    n):

        if len(features_dicts[f'user_past_ans_{n}'][user_id]) == n:
            feats_np_dic[f'rolling_mean_{n}'][num] = features_dicts[f'user_past_ans_{n}'][user_id].count('1')/n
        else:
            feats_np_dic[f'rolling_mean_{n}'][num] = np.nan


    def update_user_ans_list(self,user_id,target,
                                  features_dicts,
                                  n):
        if len(features_dicts[f'user_past_ans_{n}'][user_id]) == n:
            features_dicts[f'user_past_ans_{n}'][user_id] = features_dicts[f'user_past_ans_{n}'][user_id][1:]
            features_dicts[f'user_past_ans_{n}'][user_id] += str(target)
        else:
            features_dicts[f'user_past_ans_{n}'][user_id] += str(target)


    def update_feats(self,previous_row,features_dicts):
        # メモリ削減のため型変換
        user_id = int(previous_row[0])
        target = int(previous_row[1])
        content_id = int(previous_row[2])
        prior_question_elapsed_time = previous_row[3]
        prior_question_had_explanation = int(previous_row[4])
        timestamp = int(previous_row[5])
        bundle_id = int(previous_row[6])
        part = int(previous_row[7])
        community = int(previous_row[8])

        # rolling mean
        self.update_user_ans_list(user_id,target,
                                  features_dicts,
                                  n=10)

        self.update_user_ans_list(user_id,target,
                                  features_dicts,
                                  n=3)

    # 過去分アップデート
    def update_previous(self,features_dicts,previous_df):
        for previous_row in previous_df:
            self.update_feats(previous_row,features_dicts)



    # dataframeに格納するnpを一括作成
    def create_datas(self,df):
        df_name_float_list = [
                    # User
                    'rolling_mean_10',
                    'rolling_mean_3'
        ]

        feats_np_dic = {}
        for name in df_name_float_list:
            feats_np_dic[name] = np.zeros(len(df), dtype = np.float32)

        return feats_np_dic

    def add_past_feature(self,df, features_dicts,_update = True):
        # 特徴量格納dicを作成
        feats_np_dic = self.create_datas(df)
        previous_bundle_id = None
        previous_user_id = None
        previous_row = None
        update_cnt = 0
        previous_df = []


        for num, row in enumerate(tqdm(df[['user_id', 'answered_correctly', 'content_id', 'prior_question_elapsed_time',
                                            'prior_question_had_explanation', 'timestamp','bundle_id','part','community']].values)):
            # メモリ削減のため型変換
            user_id = int(row[0])
            target = int(row[1])
            content_id = int(row[2])
            prior_question_elapsed_time = row[3]
            prior_question_had_explanation = int(row[4])
            timestamp = int(row[5])
            bundle_id = int(row[6])
            part = int(row[7])
            community = int(row[8])

            update = _update
            # 前回とbundle_idが同じ時は更新しない
            if (previous_bundle_id == bundle_id) & (previous_user_id == user_id) & (_update):
                update = False
                if update_cnt == 0:
                    previous_df.append(previous_row)
                previous_df.append(row)
                update_cnt += 1

            # 溜まっていたら過去情報をupdate
            if (update) & (len(previous_df) > 0):
                self.update_previous(features_dicts,previous_df)
                previous_df = []
                update_cnt = 0
                update = False

            if (update) & (previous_row is not None):
                self.update_feats(previous_row,features_dicts)

            previous_bundle_id = bundle_id
            previous_user_id = user_id
            previous_row = row

            self.create_user_ans_rolling_mean(num,user_id,target,
                                              features_dicts,
                                              feats_np_dic,
                                              n=10)

            self.create_user_ans_rolling_mean(num,user_id,target,
                                              features_dicts,
                                              feats_np_dic,
                                              n=3)


        loop_feats_df = pd.DataFrame(feats_np_dic)
        df = pd.concat([df, loop_feats_df], axis = 1)
        return df,feats_np_dic.keys()

    def create_dics(self):
        features_dicts = {}
        str_name = [
                    # User rolling
                    'user_past_ans_10',
                    'user_past_ans_3'
        ]

        for name in str_name:
            features_dicts[name] = defaultdict(str)

        return features_dicts

    def create_features(self):

        self.train = pd.read_feather(f'./{Feature.dir}/BASE_FIX_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_FIX_valid.feather')
        self.train['prior_question_elapsed_time'] = self.train['prior_question_elapsed_time'].fillna(0)
        self.valid['prior_question_elapsed_time'] = self.valid['prior_question_elapsed_time'].fillna(0)

        features_dicts = self.create_dics()

        self.train , _ = self.add_past_feature(self.train, features_dicts)
        self.valid , create_feats = self.add_past_feature(self.valid, features_dicts)

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]


        with open(f'./features/all_data/loop_feats_rolling_mean2.dill','wb') as f:
            dill.dump(features_dicts,f)

class ROLLING_PART_MEAN3(Feature):

    # rolling mean
    def create_user_ans_rolling_part_mean(self,num,user_id,part,
                                          features_dicts,
                                          feats_np_dic,
                                          n):

        if len(features_dicts[f'user_past_part_ans_{n}'][user_id][part]) == n:
            feats_np_dic[f'rolling_part_mean_{n}'][num] = features_dicts[f'user_past_part_ans_{n}'][user_id][part].count('1')/n
        else:
            feats_np_dic[f'rolling_part_mean_{n}'][num] = np.nan


    def update_user_part_ans_list(self,user_id,target,part,
                                  features_dicts,
                                  n):
        if len(features_dicts[f'user_past_part_ans_{n}'][user_id][part]) == n:
            features_dicts[f'user_past_part_ans_{n}'][user_id][part] = features_dicts[f'user_past_part_ans_{n}'][user_id][part][1:]
            features_dicts[f'user_past_part_ans_{n}'][user_id][part] += str(target)
        else:
            features_dicts[f'user_past_part_ans_{n}'][user_id][part] += str(target)


    def update_feats(self,previous_row,features_dicts):
        # メモリ削減のため型変換
        user_id = int(previous_row[0])
        target = int(previous_row[1])
        content_id = int(previous_row[2])
        prior_question_elapsed_time = previous_row[3]
        prior_question_had_explanation = int(previous_row[4])
        timestamp = int(previous_row[5])
        bundle_id = int(previous_row[6])
        part = int(previous_row[7])
        community = int(previous_row[8])

        # rolling mean
        self.update_user_part_ans_list(user_id,target,part,
                                  features_dicts,
                                  n=10)

        self.update_user_part_ans_list(user_id,target,part,
                                  features_dicts,
                                  n=3)

    # 過去分アップデート
    def update_previous(self,features_dicts,previous_df):
        for previous_row in previous_df:
            self.update_feats(previous_row,features_dicts)



    # dataframeに格納するnpを一括作成
    def create_datas(self,df):
        df_name_float_list = [
                    # User
                    'rolling_part_mean_10',
                    'rolling_part_mean_3'
        ]

        feats_np_dic = {}
        for name in df_name_float_list:
            feats_np_dic[name] = np.zeros(len(df), dtype = np.float32)

        return feats_np_dic

    def add_past_feature(self,df, features_dicts,_update = True):
        # 特徴量格納dicを作成
        feats_np_dic = self.create_datas(df)
        previous_bundle_id = None
        previous_user_id = None
        previous_row = None
        update_cnt = 0
        previous_df = []


        for num, row in enumerate(tqdm(df[['user_id', 'answered_correctly', 'content_id', 'prior_question_elapsed_time',
                                            'prior_question_had_explanation', 'timestamp','bundle_id','part','community']].values)):
            # メモリ削減のため型変換
            user_id = int(row[0])
            target = int(row[1])
            content_id = int(row[2])
            prior_question_elapsed_time = row[3]
            prior_question_had_explanation = int(row[4])
            timestamp = int(row[5])
            bundle_id = int(row[6])
            part = int(row[7])
            community = int(row[8])

            update = _update
            # 前回とbundle_idが同じ時は更新しない
            if (previous_bundle_id == bundle_id) & (previous_user_id == user_id) & (_update):
                update = False
                if update_cnt == 0:
                    previous_df.append(previous_row)
                previous_df.append(row)
                update_cnt += 1

            # 溜まっていたら過去情報をupdate
            if (update) & (len(previous_df) > 0):
                self.update_previous(features_dicts,previous_df)
                previous_df = []
                update_cnt = 0
                update = False

            if (update) & (previous_row is not None):
                self.update_feats(previous_row,features_dicts)

            previous_bundle_id = bundle_id
            previous_user_id = user_id
            previous_row = row

            self.create_user_ans_rolling_part_mean(num,user_id,part,
                                              features_dicts,
                                              feats_np_dic,
                                              n=10)

            self.create_user_ans_rolling_part_mean(num,user_id,part,
                                              features_dicts,
                                              feats_np_dic,
                                              n=3)


        loop_feats_df = pd.DataFrame(feats_np_dic)
        df = pd.concat([df, loop_feats_df], axis = 1)
        return df,feats_np_dic.keys()

    def create_dics(self):
        features_dicts = {}
        str_name = [
                    # User rolling
                    'user_past_part_ans_10',
                    'user_past_part_ans_3'
        ]

        for name in str_name:
            features_dicts[name] = defaultdict(lambda: defaultdict(str))

        return features_dicts

    def create_features(self):

        self.train = pd.read_feather(f'./{Feature.dir}/BASE_FIX_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_FIX_valid.feather')
        self.train['prior_question_elapsed_time'] = self.train['prior_question_elapsed_time'].fillna(0)
        self.valid['prior_question_elapsed_time'] = self.valid['prior_question_elapsed_time'].fillna(0)

        features_dicts = self.create_dics()

        self.train , _ = self.add_past_feature(self.train, features_dicts)
        self.valid , create_feats = self.add_past_feature(self.valid, features_dicts)

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]


        with open(f'./features/all_data/loop_feats_rolling_part_mean3.dill','wb') as f:
            dill.dump(features_dicts,f)


# class QUESTION(Feature):

#     def create_features(self):
#         create_feats = [f'tag_{int(i)}' for i in range(188)]
#         self.train = pd.read_feather(f'./{Feature.dir}/BASE_FIX_train.feather')
#         self.valid = pd.read_feather(f'./{Feature.dir}/BASE_FIX_valid.feather')

#         questions = pd.read_csv('./data/input/questions.csv')
#         lst = []
#         for tags in questions["tags"]:
#             ohe = np.zeros(188)
#             if str(tags) != "nan":
#                 for tag in tags.split():
#                     ohe += np.eye(188)[int(tag)]
#             lst.append(ohe)
#         tags_df = pd.DataFrame(lst, columns=[f"tag_{i}" for i in range(188)]).astype(int)

#         questions = pd.concat([questions,tags_df],axis=1)
#         questions = questions.rename(columns={'question_id':'content_id'})

#         self.train = pd.merge(self.train,questions,on='content_id',how='left')
#         self.valid = pd.merge(self.valid,questions,on='content_id',how='left')

#         self.train = self.train[create_feats]
#         self.valid = self.valid[create_feats]



if __name__ == "__main__":
    args = get_arguments()
    generate_features(globals(),args.force)