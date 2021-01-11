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


class LOOP_LAST(Feature):
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


        # lag time
        self.update_lag_incorrect_feats(user_id,timestamp,target,
                                        features_dicts)

        self.update_tags_feats(tags_ls,
                               target,
                               features_dicts,
                               n=4)

        # User args feats
        create_lists = [[part,
                        'user_part_list',                                                       # dic
                        'user_part_count','ans_user_part_avg']                                 # np
                        ]
        for create_list in create_lists:
            self.update_user_arg_feats(user_id,create_list[0],target,
                                                        features_dicts,
                                                        create_list[1])

        # arg feats
        create_lists = [[user_id,
                        'user_list'],
                        [content_id,
                        'content_list']]
        for create_list in create_lists:
            self.update_args_feats(create_list[0],create_list[1],
                                  target,
                                  features_dicts,
                                  n=4)


        if time_update:
            self.update_lag_time_feats(user_id,timestamp,
                                    features_dicts)

            self.update_part_lag_time_feats(user_id,part,timestamp,
                                            features_dicts)

            self.update_tags_time_feats(tags_ls,prior_question_elapsed_time,prior_question_had_explanation,
                                        features_dicts)

            create_lists = [[user_id,
                            'user_list'],
                            [content_id,
                            'content_list']]
            for create_list in create_lists:
                self.update_args_time_feats(create_list[0],prior_question_elapsed_time,prior_question_had_explanation,
                                features_dicts,
                                create_list[1])
        # rolling mean
        self.update_user_ans_list(user_id,target,
                                  features_dicts,
                                  n=10)

        self.update_user_ans_list(user_id,target,
                                  features_dicts,
                                  n=3)

        self.update_user_part_ans_list(user_id,target,part,
                                  features_dicts,
                                  n=10)

        self.update_user_part_ans_list(user_id,target,part,
                                  features_dicts,
                                  n=3)

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
                    # rolling mean
                    'rolling_mean_10',
                    'rolling_mean_3',
                    # rolling part mean
                    'rolling_part_mean_10',
                    'rolling_part_mean_3'
        ]

        df_name_int_list = [
                    # User Content
                    'user_content_count',
                    # User Part
                    'user_part_count'
        ]

        tags_mean_name = [f'tags_{i+1}_mean' for i in range(6)]
        tags_elapsed_time = [f'tags_{i+1}_elapsed_time_avg' for i in range(6)]
        tags_explanation_name = [f'tags_{i+1}_explanation_avg' for i in range(6)]

        df_name_float_list = df_name_float_list + tags_mean_name + tags_elapsed_time + tags_explanation_name


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


        for num, row in enumerate(df[['user_id', 'answered_correctly', 'content_id', 'prior_question_elapsed_time',
                                            'prior_question_had_explanation', 'timestamp','bundle_id','part','community','tags_ls']].values):
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


            # Args
            create_lists = [[user_id,
                            'user_list',                 # dic
                            'ans_user_avg','elapsed_time_user_avg','explanation_user_avg','elapsed_time_user_sum'],                                 # np
                            [content_id,
                            'content_list',     # dic
                            'ans_content_avg','elapsed_time_content_avg','explanation_content_avg',None]                        # np
                            ]


            for create_list in create_lists:
                self.create_arg_feats(num,create_list[0],
                                    features_dicts,feats_np_dic,
                                    create_list[1],
                                    create_list[2],create_list[3],create_list[4],create_list[5])

                # # 常に更新
                # self.update_args_time_feats(create_list[0],prior_question_elapsed_time,prior_question_had_explanation,
                #                             features_dicts,
                #                             create_list[1])

            # First bundle
            self.create_first_bundle(num,user_id,bundle_id,
                                    features_dicts,
                                    feats_np_dic)


            # lag time
            self.create_lag_time_feats(num,user_id,timestamp,
                                    features_dicts,
                                    feats_np_dic)

            self.create_part_lag_time_feats(num,user_id,part,timestamp,
                                            features_dicts,
                                            feats_np_dic)

            # rolling mean
            self.create_user_ans_rolling_mean(num,user_id,target,
                                              features_dicts,
                                              feats_np_dic,
                                              n=10)

            self.create_user_ans_rolling_mean(num,user_id,target,
                                              features_dicts,
                                              feats_np_dic,
                                              n=3)

            self.create_user_ans_rolling_part_mean(num,user_id,part,
                                              features_dicts,
                                              feats_np_dic,
                                              n=10)

            self.create_user_ans_rolling_part_mean(num,user_id,part,
                                              features_dicts,
                                              feats_np_dic,
                                              n=3)

            self.create_tags_feats(num,tags_ls,
                                   features_dicts,feats_np_dic)

            # # rolling mean
            # self.create_user_ans_rolling_mean(num,user_id,target,
            #                                   features_dicts,
            #                                   feats_np_dic)


            # User args feats
            # ------------------------------------------------------------------
            create_lists = [[part,
                            'user_part_list',                                                       # dic
                            'user_part_count','ans_user_part_avg'],                                 # np
                            ]
            for create_list in create_lists:
                self.create_user_args_feats(num,user_id,create_list[0],
                                            features_dicts,feats_np_dic,
                                            create_list[1],
                                            create_list[2],create_list[3]
                                            )



            # TODO:Updateに回すべきか
            # User args count
            # ------------------------------------------------------------------
            create_lists = [[content_id,
                            'user_content_count',                            # dic
                            'user_content_count']                            # np
                            ]
            for create_list in create_lists:
                self.update_create_user_arg_count(num,user_id,create_list[0],
                                                  features_dicts,feats_np_dic,
                                                  create_list[1],
                                                  create_list[2])

            # TODO : カウントは毎回更新するべきか
            # User count
            feats_np_dic['ans_user_count'][num] = features_dicts['user_list'][user_id][0]


            # Update
            # ------------------------------------------------------------------
            # lag time
            # ------------------------------------------------------------------
            # self.update_lag_time_feats(user_id,timestamp,
            #                         features_dicts)

            # self.update_part_lag_time_feats(user_id,part,timestamp,
            #                                 features_dicts)

            # count
            # ------------------------------------------------------------------
            # features_dicts['ans_user_count'][user_id] += 1
            # features_dicts['ans_content_count'][content_id] += 1



        loop_feats_df = pd.DataFrame(feats_np_dic)

        tags_mean_name = [f'tags_{i+1}_mean' for i in range(6)]
        tags_elapsed_time = [f'tags_{i+1}_elapsed_time_avg' for i in range(6)]
        tags_explanation_name = [f'tags_{i+1}_explanation_avg' for i in range(6)]

        loop_feats_df['tags_mean'] = loop_feats_df[tags_mean_name].mean(axis=1)
        loop_feats_df['tags_elapsed_time_avg'] = loop_feats_df[tags_elapsed_time].mean(axis=1)
        loop_feats_df['tags_explanation_avg'] = loop_feats_df[tags_explanation_name].mean(axis=1)


        tags_mean_name = [f'tags_{i+1}_mean' for i in range(6)]
        tags_elapsed_time = [f'tags_{i+1}_elapsed_time_avg' for i in range(6)]
        tags_explanation_name = [f'tags_{i+1}_explanation_avg' for i in range(6)]

        loop_feats_df['tags_mean'] = loop_feats_df[tags_mean_name].mean(axis=1)
        loop_feats_df['tags_elapsed_time_avg'] = loop_feats_df[tags_elapsed_time].mean(axis=1)

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
                    # tags1
                    'tags_list',
                    # User Time
                    'lag_user_time',
                    'lag_user_incorrect_time'
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
                    # User Part
                    'user_part_list'
        ]

        str_lambda_name = ['user_past_part_ans_10',
                           'user_past_part_ans_3']

        str_name = ['user_past_ans_10',
                    'user_past_ans_3']

        for name in str_name:
            features_dicts[name] = defaultdict(str)

        for name in str_lambda_name:
            features_dicts[name] = defaultdict(lambda: defaultdict(str))

        for name in list_name:
            features_dicts[name] = defaultdict(list)


        for name in lambda_int_name:
            features_dicts[name] = defaultdict(lambda: defaultdict(int))

        for name in lambda_list_name:
            features_dicts[name] = defaultdict(lambda: defaultdict(list))


        return features_dicts


    def create_features(self):

        self.train = pd.read_feather(f'./{Feature.dir}/BASE_FIX_train.feather')[:10000]
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_FIX_valid.feather')[:10000]
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


        with open(f'./features/all_data/loop_feats_last.dill','wb') as f:
            dill.dump(features_dicts,f)



if __name__ == "__main__":
    args = get_arguments()
    generate_features(globals(),args.force)