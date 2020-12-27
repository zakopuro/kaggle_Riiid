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
if MINI_DATA == True:
    Feature.dir = 'features/kernel_mini_data'
    CV = 'cv30'
    print('MINI DATA')
else:
    Feature.dir = 'features/data'
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
        train_index = pd.read_feather(f'./data/train_valid/{CV}_train.feather')
        valid_index = pd.read_feather(f'./data/train_valid/{CV}_valid.feather')

        qs = pd.read_csv('./data/input/questions.csv')
        lc = pd.read_csv('./data/input/lectures_new.csv')
        tag = qs["tags"].str.split(" ",expand = True)
        tag.columns = ['tags1','tags2','tags3','tags4','tags5','tags6']
        qs = pd.concat([qs,tag],axis=1)
        lc['l_type_of'] = _label_encoder(lc['type_of'])
        qs = qs.rename(columns={'question_id':'content_id'})
        lc = lc.rename(columns={'lecture_id':'content_id'})

        qs_train = self.train[self.train['content_type_id'] == 0]
        lc_train = self.train[self.train['content_type_id'] == 1]
        qs_train = pd.merge(qs_train,qs,on='content_id',how='left')
        lc_train = pd.merge(lc_train,lc,on='content_id',how='left')

        self.train = pd.concat([qs_train,lc_train])
        self.train = self.train.sort_values('row_id')

        del qs_train,lc_train
        gc.collect()



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


        self.train = self.train[self.train['content_type_id'] == 0].reset_index(drop=True)
        self.valid = self.valid[self.valid['content_type_id'] == 0].reset_index(drop=True)
        # self.test = self.test[self.test['content_type_id'] == 0].reset_index(drop=True)

        prior_question_elapsed_time_mean = self.train.prior_question_elapsed_time.dropna().values.mean()
        self.train['prior_question_elapsed_time_mean'] = self.train.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)
        self.valid['prior_question_elapsed_time_mean'] = self.valid.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)


        self.test = pd.DataFrame()

class GROUP_BY(Feature):

    def create_features(self):
        create_feats = ['task_container_count','paid_user_part_mean']
        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')

        self.train['count'] = 1
        self.valid['count'] = 1

        train_task_container_count = self.train[['user_id','task_container_id','count']].groupby(['user_id','task_container_id']).agg(['count']).reset_index()
        valid_task_container_count = self.valid[['user_id','task_container_id','count']].groupby(['user_id','task_container_id']).agg(['count']).reset_index()
        train_task_container_count.columns = ['user_id','task_container_id','task_container_count']
        valid_task_container_count.columns = ['user_id','task_container_id','task_container_count']

        self.train = pd.merge(self.train,train_task_container_count,on=['user_id','task_container_id'],how='left')
        self.valid = pd.merge(self.valid,train_task_container_count,on=['user_id','task_container_id'],how='left')


        train_paid_df = pd.read_feather(f'./{Feature.dir}/PAID_USER_train.feather')
        valid_paid_df = pd.read_feather(f'./{Feature.dir}/PAID_USER_valid.feather')

        self.train = pd.concat([self.train,train_paid_df],axis=1)
        self.valid = pd.concat([self.valid,valid_paid_df],axis=1)

        paid_user_part_mean = self.train[[TARGET,'paid_user','part']].groupby(['paid_user','part']).agg(['mean']).reset_index()
        paid_user_part_mean.columns = ['paid_user','part','paid_user_part_mean']
        self.train = pd.merge(self.train,paid_user_part_mean,on=['part','paid_user'],how='left')
        self.valid = pd.merge(self.valid,paid_user_part_mean,on=['part','paid_user'],how='left')

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]


class PAID_USER(Feature):

    # 有料会員を特定する
    def create_features(self):
        create_feats = ['paid_user']
        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')

        self.train['days_elapsed'] = self.train['timestamp']/(1000*3600*24)
        self.valid['days_elapsed'] = self.valid['timestamp']/(1000*3600*24)
        self.train['days_elapsed'] = self.train['days_elapsed'].astype(int)
        self.valid['days_elapsed'] = self.valid['days_elapsed'].astype(int)


        df = pd.concat([self.train,self.valid]).reset_index(drop=True)

        df['count'] = 1
        user_day_count = df[df['days_elapsed'] >= 1].groupby(['user_id','days_elapsed'])['count'].agg(['count'])
        user_day_count = user_day_count.reset_index()
        # 1日50問以上解いているユーザーを有料会員にする
        paid_user = user_day_count[user_day_count['count'] > 50]['user_id'].unique()

        df['paid_user'] = 0
        df.loc[df['user_id'].isin(paid_user),'paid_user'] = 1
        self.train = df[:len(self.train)]
        self.valid = df[len(self.train):]
        self.train = self.train.reset_index(drop=True)
        self.valid = self.valid.reset_index(drop=True)

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]



class TAGS(Feature):

    def create_features(self):
        create_feats = ["tags_pca_0", "tags_pca_1"]
        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')

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

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]

        pca_feat_df.to_feather(f'./{Feature.dir}/pca_tags.feather')





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


    def create_first_bundle(self,num,user_id,bundle_id,
                            features_dicts,
                            feats_np_dic):

        if features_dicts['user_first_bundle'][user_id] == 0:
            features_dicts['user_first_bundle'][user_id] = bundle_id

        feats_np_dic['first_bundle'][num] = features_dicts['user_first_bundle'][user_id]

    def update_create_user_arg_count(self,num,user_id,col,
                                     features_dicts,feats_np_dic,
                                     user_args_count_dic_name,
                                     user_args_count_name):

        feats_np_dic[user_args_count_name][num] = features_dicts[user_args_count_dic_name][user_id][col]

        # update
        features_dicts[user_args_count_dic_name][user_id][col] += 1



    def update_previous(self,row,features_dicts,previous_df):
        for row in previous_df:
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
            bundle_id = int(row[8])



            # lag time
            self.update_lag_incorrect_feats(user_id,timestamp,target,
                                        features_dicts)

            # part lag time
            self.update_part_lag_incorrect_feats(user_id,part,timestamp,target,
                                            features_dicts)
            # args feats
            # ------------------------------------------------------------------
            create_lists = [[user_id,
                    'ans_user_count','ans_user_sum','elapsed_time_user_sum','explanation_user_sum',                 # dic
                    'ans_user_avg','elapsed_time_user_avg','explanation_user_avg'],                                 # np
                    [content_id,
                    'ans_content_count','ans_content_sum','elapsed_time_content_sum','explanation_content_sum',     # dic
                    'ans_content_avg','elapsed_time_content_avg','explanation_content_avg'],                        # np
                    [tags1,
                    'ans_tags1_count','ans_tags1_sum','elapsed_time_tags1_sum','explanation_tags1_sum',             # dic
                    'ans_tags1_avg','elapsed_time_tags1_avg','explanation_tags1_avg']                               # np
                    ]
            for create_list in create_lists:
                self.update_arg_feats(create_list[0],target,
                                                        features_dicts,
                                                        create_list[1],create_list[2])

            # User args feats
            # ------------------------------------------------------------------
            create_lists = [[part,
                            'ans_user_part_list',                                                       # dic
                            'ans_user_part_count','ans_user_part_avg'],                                 # np
                            ]
            for create_list in create_lists:
                self.update_user_arg_feats(user_id,create_list[0],target,
                                        features_dicts,
                                        create_list[1])

    # 特徴量アップデート
    def update_feats(self,previous_row,features_dicts):
        user_id = int(previous_row[0])
        target = int(previous_row[1])
        content_id = int(previous_row[2])
        # prior_question_elapsed_time = previous_row[3]
        # prior_question_had_explanation = int(previous_row[4])
        timestamp = int(previous_row[5])
        # Nanのときはそのまま(float)にする
        try:
            tags1 = int(previous_row[6])
        except:
            tags1 = previous_row[6]

        part = int(previous_row[7])
        bundle_id = int(previous_row[8])

        self.update_lag_incorrect_feats(user_id,timestamp,target,
                                        features_dicts)

        self.update_part_lag_incorrect_feats(user_id,part,timestamp,target,
                                                    features_dicts)


        create_lists = [[user_id,
                        'ans_user_count','ans_user_sum','elapsed_time_user_sum','explanation_user_sum',                 # dic
                        'ans_user_avg','elapsed_time_user_avg','explanation_user_avg'],                                 # np
                        [content_id,
                        'ans_content_count','ans_content_sum','elapsed_time_content_sum','explanation_content_sum',     # dic
                        'ans_content_avg','elapsed_time_content_avg','explanation_content_avg'],                        # np
                        [tags1,
                        'ans_tags1_count','ans_tags1_sum','elapsed_time_tags1_sum','explanation_tags1_sum',             # dic
                        'ans_tags1_avg','elapsed_time_tags1_avg','explanation_tags1_avg']                               # np
                        ]

        for create_list in create_lists:
            self.update_arg_feats(create_list[0],target,
                    features_dicts,
                    create_list[1],create_list[2])

        create_lists = [[part,
                        'ans_user_part_list',                                                       # dic
                        'ans_user_part_count','ans_user_part_avg'],                                 # np
                        ]

        for create_list in create_lists:
            self.update_user_arg_feats(user_id,create_list[0],target,
                                                        features_dicts,
                                                        create_list[1])







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
                    # Tags1
                    'ans_tags1_avg',
                    'elapsed_time_tags1_avg',
                    'explanation_tags1_avg',
                    # User Part
                    'ans_user_part_avg',
                    # User Part lag time
                    'lag_part_time_1',
                    'lag_part_time_2',
                    'lag_part_time_3',
                    'lag_part_incorrect_time',
                    # User First Bundle
                    'first_bundle'
        ]

        df_name_int_list = [
                    # Usr Part
                    'ans_user_part_count',
                    # User Content
                    'user_content_count'
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

        for num, row in enumerate(tqdm(df[['user_id', 'answered_correctly', 'content_id',
                                           'prior_question_elapsed_time', 'prior_question_had_explanation', 'timestamp',
                                           'tags1','part','bundle_id']].values)):
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
            bundle_id = int(row[8])


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
                self.update_previous(row,features_dicts,previous_df)
                previous_df = []
                update_cnt = 0


            if (update) & (previous_row is not None):
                self.update_feats(previous_row,features_dicts)



            previous_bundle_id = bundle_id
            previous_user_id = user_id
            previous_row = row

            # lag time
            # ------------------------------------------------------------------
            self.create_lag_time_feats(num,user_id,timestamp,
                                    features_dicts,
                                    feats_np_dic)
            # 更新
            self.update_lag_time_feats(user_id,timestamp,
                                    features_dicts)
            # if update:
            #     self.update_lag_incorrect_feats(user_id,timestamp,target,
            #                                     features_dicts)


            # Part lag time
            # ------------------------------------------------------------------
            self.create_part_lag_time_feats(num,user_id,part,timestamp,
                                            features_dicts,
                                            feats_np_dic)
            # 更新
            self.update_part_lag_time_feats(user_id,part,timestamp,
                                            features_dicts)
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
                            'ans_content_avg','elapsed_time_content_avg','explanation_content_avg'],                        # np
                            [tags1,
                            'ans_tags1_count','ans_tags1_sum','elapsed_time_tags1_sum','explanation_tags1_sum',             # dic
                            'ans_tags1_avg','elapsed_time_tags1_avg','explanation_tags1_avg']                               # np
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
                # if update:
                    # self.update_arg_feats(create_list[0],target,
                    #                     features_dicts,
                    #                     create_list[1],create_list[2])

            # User args feats
            # ------------------------------------------------------------------
            create_lists = [[part,
                            'ans_user_part_list',                                                       # dic
                            'ans_user_part_count','ans_user_part_avg'],                                 # np
                            ]


            for create_list in create_lists:
                self.create_user_args_feats(num,user_id,create_list[0],
                                            features_dicts,feats_np_dic,
                                            create_list[1],
                                            create_list[2],create_list[3]
                                            )

                # if update:
                #     self.update_user_arg_feats(user_id,create_list[0],target,
                #                                features_dicts,
                #                                create_list[1])

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


            # First bundle
            # ------------------------------------------------------------------
            self.create_first_bundle(num,user_id,bundle_id,
                                    features_dicts,
                                    feats_np_dic)



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

        # User First Bundle
        user_first_bundle_dic = defaultdict(int)

        # User content count
        user_content_count_dic = defaultdict(lambda: defaultdict(int))

        # {'User id':{'XXX(特徴量名)':[1(カウント数),1(正解数)]}} とする
        # User Part
        ans_user_part_list_dic = defaultdict(lambda: defaultdict(list))

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
                        # Tags1
                        'ans_tags1_count' : ans_tags1_count_dic,
                        'ans_tags1_sum' : ans_tags1_sum_dic,
                        'elapsed_time_tags1_sum' : elapsed_time_tags1_sum_dic,
                        'explanation_tags1_sum' : explanation_tags1_sum_dic,
                        # User Time
                        'lag_user_time' : lag_user_time_dic,
                        'lag_user_incorrect_time' : lag_user_incorrect_time_dic,
                        # User Part
                        'ans_user_part_list' : ans_user_part_list_dic,
                        # User Part Time
                        'lag_user_part_time' : lag_user_part_time_dic,
                        'lag_user_part_incorrect_time' : lag_user_part_incorrect_time_dic,
                        # User First Bundle
                        'user_first_bundle' : user_first_bundle_dic,
                        # User content count
                        'user_content_count' : user_content_count_dic
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


        with open(f'./features/kernel_mini_data/loop_feats_mini.dill','wb') as f:
            dill.dump(features_dicts,f)






if __name__ == "__main__":
    args = get_arguments()
    generate_features(globals(),args.force)