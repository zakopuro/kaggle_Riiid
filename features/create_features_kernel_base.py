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


MINI_DATA = True
Feature.dir = 'features/kernel_base'
CV = 'cv1'

Feature.kernel_dir = 'features/kernel_data'

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

class BASE(Feature):

    def create_features(self):
        self.train = pd.read_feather('./data/input/train.feather')
        # train_index = pd.read_feather(f'./data/train_valid/{CV}_train.feather')
        # valid_index = pd.read_feather(f'./data/train_valid/{CV}_valid.feather')
        train_index = pd.read_csv(f'./data/train_valid/train_rows.csv')
        valid_index = pd.read_csv(f'./data/train_valid/valid_rows.csv')


        qs = pd.read_csv('./data/input/questions.csv')
        tag = qs["tags"].str.split(" ",expand = True)
        tag.columns = ['tags1','tags2','tags3','tags4','tags5','tags6']
        qs = pd.concat([qs,tag],axis=1)
        qs = qs.rename(columns={'question_id':'content_id'})

        self.train = pd.merge(self.train,qs,on='content_id',how='left')
        self.train = self.train.sort_values('row_id')

        self.train['prior_question_had_explanation'] = self.train['prior_question_had_explanation'].fillna(False)
        self.train.loc[self.train['prior_question_had_explanation'] == False , 'prior_question_had_explanation'] = 0
        self.train.loc[self.train['prior_question_had_explanation'] == True , 'prior_question_had_explanation'] = 1
        self.train['prior_question_had_explanation'] = self.train['prior_question_had_explanation'].astype('int8')


        for i in range(1,7):
            self.train[f'tags{i}'] = self.train[f'tags{i}'].astype(float)

        self.train['small_quenstion'] = self.train['content_id'] - self.train['bundle_id']

        # self.test = self.train.copy()
        self.valid = self.train[self.train['row_id'].isin(valid_index['row_id'])]
        self.train = self.train[self.train['row_id'].isin(train_index['row_id'])]




        # self.train = self.train.reset_index(drop=True)
        # self.train = self.train.iloc[-40000000:]

        self.train = self.train[self.train['content_type_id'] == 0].reset_index(drop=True)
        self.valid = self.valid[self.valid['content_type_id'] == 0].reset_index(drop=True)
        # self.test = self.test[self.test['content_type_id'] == 0].reset_index(drop=True)

        prior_question_elapsed_time_mean = self.train.prior_question_elapsed_time.dropna().mean()
        self.train['prior_question_elapsed_time_mean'] = self.train.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)
        self.valid['prior_question_elapsed_time_mean'] = self.valid.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)


        self.test = pd.DataFrame()




class LOOP(Feature):


    def update_part_lag_time_feats(self,user_id,part,timestamp,
                                  features_dicts):
        if len(features_dicts['lag_user_part_time'][user_id][part]) == 3:
            features_dicts['lag_user_part_time'][user_id][part].pop(0)
            features_dicts['lag_user_part_time'][user_id][part].append(timestamp)
        else:
            features_dicts['lag_user_part_time'][user_id][part].append(timestamp)

    def update_part_lag_incorrect_feats(self,user_id,part,timestamp,target,
                                        features_dicts):

        if target == 0:
            if len(features_dicts['lag_user_part_incorrect_time'][user_id][part]) == 1:
                features_dicts['lag_user_part_incorrect_time'][user_id][part].pop(0)
                features_dicts['lag_user_part_incorrect_time'][user_id][part].append(timestamp)
            else:
                features_dicts['lag_user_part_incorrect_time'][user_id][part].append(timestamp)

    #
    def update_lag_time_feats(self,user_id,timestamp,
                             features_dicts):

        if len(features_dicts['lag_user_time'][user_id]) == 3:
            features_dicts['lag_user_time'][user_id].pop(0)
            features_dicts['lag_user_time'][user_id].append(timestamp)
        else:
            features_dicts['lag_user_time'][user_id].append(timestamp)

    def update_lag_incorrect_feats(self,user_id,timestamp,target,
                                   features_dicts):

        if target == 0:
            if len(features_dicts['lag_user_incorrect_time'][user_id]) == 1:
                features_dicts['lag_user_incorrect_time'][user_id].pop(0)
                features_dicts['lag_user_incorrect_time'][user_id].append(timestamp)
            else:
                features_dicts['lag_user_incorrect_time'][user_id].append(timestamp)


    # User 組み合わせ特徴量更新
    def update_user_arg_feats(self,user_id,col,target,
                              features_dicts,
                              ans_user_args_list_name
                            ):

        if len(features_dicts[ans_user_args_list_name][user_id][col]) == 0:
            features_dicts[ans_user_args_list_name][user_id][col] = [0,0]
        features_dicts[ans_user_args_list_name][user_id][col][1] += target
        features_dicts[ans_user_args_list_name][user_id][col][0] += 1

    # 引数特徴量更新
    def update_arg_feats(self,col,target,
                         features_dicts,
                         ans_args_count_name, ans_args_sum_name
                         ):

        features_dicts[ans_args_count_name][col] += 1
        features_dicts[ans_args_sum_name][col] += target

    # こっちは常に更新
    def update_args_time_feats(self,col,prior_question_elapsed_time,prior_question_had_explanation,
                              features_dicts,
                               elapsed_time_args_sum_name, explanation_args_sum_name):

        features_dicts[elapsed_time_args_sum_name][col] += prior_question_elapsed_time
        features_dicts[explanation_args_sum_name][col] += prior_question_had_explanation


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



    def create_arg_feats(self,num,col,
                        features_dicts,feats_np_dic,
                        ans_count_name = None,ans_sum_name = None, elapsed_time_sum_name = None, explanation_sum_name = None,
                        ans_avg_name = None, elapsed_time_avg_name = None, explanation_avg_name = None):

        if features_dicts[ans_count_name][col] != 0:
            feats_np_dic[ans_avg_name][num] = features_dicts[ans_sum_name][col] / features_dicts[ans_count_name][col]
            feats_np_dic[elapsed_time_avg_name][num] = features_dicts[elapsed_time_sum_name][col] / features_dicts[ans_count_name][col]
            feats_np_dic[explanation_avg_name][num] = features_dicts[explanation_sum_name][col] / features_dicts[ans_count_name][col]
        else:
            feats_np_dic[ans_avg_name][num] = np.nan
            feats_np_dic[elapsed_time_avg_name][num] = np.nan
            feats_np_dic[explanation_avg_name][num] = np.nan



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


    # User part lag time
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

        if len(features_dicts['lag_user_part_incorrect_time'][user_id][part]) == 0:
            feats_np_dic['lag_part_incorrect_time'][num] = np.nan
        else:
            feats_np_dic['lag_part_incorrect_time'][num] = timestamp - features_dicts['lag_user_part_incorrect_time'][user_id][part][0]




    # dataframeに格納するnpを一括作成
    def create_datas(self,df):
        df_name_float_list = [
                    # User
                    'ans_user_avg',
                    'elapsed_time_user_avg',
                    'explanation_user_avg',
                    # content_id
                    'ans_content_avg',
                    'elapsed_time_content_avg',
                    'explanation_content_avg',
                    # user lag time
                    'lag_time_1',
                    'lag_time_2',
                    'lag_time_3',
                    'lag_incorrect_time',

                    'ans_user_content_avg'# 使わない
        ]

        df_name_int_list = [
                    # User content
                    'ans_user_content_count'
        ]

        feats_np_dic = {}
        for name in df_name_float_list:
            feats_np_dic[name] = np.zeros(len(df), dtype = np.float32)
        for name in df_name_int_list:
            feats_np_dic[name] = np.zeros(len(df), dtype = np.int32)

        return feats_np_dic

    def add_past_feature(self,df, features_dicts,update = True):

        # 特徴量格納dicを作成
        feats_np_dic = self.create_datas(df)

        for num, row in enumerate(tqdm(df[['user_id', 'answered_correctly', 'content_id',
                                           'prior_question_elapsed_time', 'prior_question_had_explanation', 'timestamp',
                                           'tags1','part']].values)):
            # メモリ削減のため型変換
            user_id = int(row[0])
            target = int(row[1])
            content_id = int(row[2])
            prior_question_elapsed_time = row[3]
            prior_question_had_explanation = int(row[4])
            timestamp = int(row[5])
            # Nanのときはそのまま(float)にする
            try:
                tags1 = int(row[6])
            except:
                tags1 = row[6]

            part = int(row[7])



            # lag time
            # ------------------------------------------------------------------
            self.create_lag_time_feats(num,user_id,timestamp,
                                    features_dicts,
                                    feats_np_dic)
            # 更新
            self.update_lag_time_feats(user_id,timestamp,
                                    features_dicts)
            if update:
                self.update_lag_incorrect_feats(user_id,timestamp,target,
                                                features_dicts)


            # Part lag time
            # ------------------------------------------------------------------
            # self.create_part_lag_time_feats(num,user_id,part,timestamp,
            #                                 features_dicts,
            #                                 feats_np_dic)
            # # 更新
            # self.update_part_lag_time_feats(user_id,part,timestamp,
            #                                 features_dicts)
            # if update:
            #     self.update_part_lag_incorrect_feats(user_id,part,timestamp,target,
            #                                         features_dicts)

            # args feats
            # ------------------------------------------------------------------
            create_lists = [[user_id,
                            'ans_user_count','ans_user_sum','elapsed_time_user_sum','explanation_user_sum',                 # dic
                            'ans_user_avg','elapsed_time_user_avg','explanation_user_avg'],                                 # np
                            [content_id,
                            'ans_content_count','ans_content_sum','elapsed_time_content_sum','explanation_content_sum',     # dic
                            'ans_content_avg','elapsed_time_content_avg','explanation_content_avg']                         # np
                            ]

            for create_list in create_lists:
                self.create_arg_feats(num,create_list[0],
                                    features_dicts,feats_np_dic,
                                    create_list[1],create_list[2],create_list[3],create_list[4],
                                    create_list[5],create_list[6],create_list[7])

                # 常に更新
                self.update_args_time_feats(create_list[0],prior_question_elapsed_time,prior_question_had_explanation,
                                            features_dicts,
                                            create_list[3],create_list[4])

                # update時のみ
                if update:
                    self.update_arg_feats(create_list[0],target,
                                        features_dicts,
                                        create_list[1],create_list[2])

            # User args feats
            # ------------------------------------------------------------------
            create_lists = [[content_id,
                            'ans_user_content_list',                                                    # dic
                            'ans_user_content_count','ans_user_content_avg']                            # np
                            ]

            for create_list in create_lists:
                self.create_user_args_feats(num,user_id,create_list[0],
                                            features_dicts,feats_np_dic,
                                            create_list[1],
                                            create_list[2],create_list[3]
                                            )

                # if update:
                self.update_user_arg_feats(user_id,create_list[0],target,
                                            features_dicts,
                                            create_list[1])


        loop_feats_df = pd.DataFrame(feats_np_dic)

        df = pd.concat([df, loop_feats_df], axis = 1)
        return df,feats_np_dic.keys()



    def create_dics(self):

        # User
        ans_user_count_dic = defaultdict(int)
        ans_user_sum_dic = defaultdict(int)
        elapsed_time_user_sum_dic = defaultdict(int)
        explanation_user_sum_dic = defaultdict(int)

        # Content_id
        ans_content_count_dic = defaultdict(int)
        ans_content_sum_dic = defaultdict(int)
        elapsed_time_content_sum_dic = defaultdict(int)
        explanation_content_sum_dic = defaultdict(int)

        # Tags1
        ans_tags1_count_dic = defaultdict(int)
        ans_tags1_sum_dic = defaultdict(int)
        elapsed_time_tags1_sum_dic = defaultdict(int)
        explanation_tags1_sum_dic = defaultdict(int)

        # User Time
        lag_user_time_dic = defaultdict(list)
        lag_user_incorrect_time_dic = defaultdict(list)


        # {'User id':{'XXX(特徴量名)':[1(カウント数),1(正解数)]}} とする
        # User content_id
        ans_user_content_list_dic = defaultdict(lambda: defaultdict(list))

        # User Part
        ans_user_part_list_dic = defaultdict(lambda: defaultdict(list))

        # User Tags1
        ans_user_tags1_list_dic = defaultdict(lambda: defaultdict(list))

        # User Part Time
        lag_user_part_time_dic = defaultdict(lambda: defaultdict(list))
        lag_user_part_incorrect_time_dic = defaultdict(lambda: defaultdict(list))

        features_dicts = {
                        # User
                        'ans_user_count' : ans_user_count_dic,
                        'ans_user_sum' : ans_user_sum_dic,
                        'elapsed_time_user_sum' : elapsed_time_user_sum_dic,
                        'explanation_user_sum' : explanation_user_sum_dic,
                        # content_id
                        'ans_content_count' : ans_content_count_dic,
                        'ans_content_sum' : ans_content_sum_dic,
                        'elapsed_time_content_sum' : elapsed_time_content_sum_dic,
                        'explanation_content_sum' : explanation_content_sum_dic,

                        # User Time
                        'lag_user_time' : lag_user_time_dic,
                        'lag_user_incorrect_time' : lag_user_incorrect_time_dic,
                        # User content_id
                        'ans_user_content_list' : ans_user_content_list_dic
        }

        return features_dicts


    def create_features(self):

        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')

        features_dicts = self.create_dics()

        self.train , _ = self.add_past_feature(self.train, features_dicts)
        self.valid , create_feats = self.add_past_feature(self.valid, features_dicts)

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]


        with open(f'./features/kernel_base/loop_feats_mini.dill','wb') as f:
            dill.dump(features_dicts,f)











if __name__ == "__main__":
    args = get_arguments()
    generate_features(globals(),args.force)