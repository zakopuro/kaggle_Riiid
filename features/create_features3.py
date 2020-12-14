import pandas as pd
import numpy as np
import gc
import os
from tqdm import tqdm
from collections import defaultdict
from base_create_features import Feature, get_arguments, generate_features
from pathlib import Path
import pickle
import sys
sys.path.append('./')
from utils import data_util,logger

MINI_DATA = False
if MINI_DATA == True:
    Feature.dir = 'features/mini_data'
    CV = 'cv37'
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



class LOOP(Feature):

    def user_feats_update(self,num,row,
                         answered_correctly_u_count_dic,answered_correctly_u_sum_dic,elapsed_time_u_sum_dic,explanation_u_sum_dic, # dic
                         answered_correctly_u_avg,elapsed_time_u_avg,explanation_u_avg # df
                         ):


        if answered_correctly_u_count_dic[row[0]] != 0:
            answered_correctly_u_avg[num] = answered_correctly_u_sum_dic[row[0]] / answered_correctly_u_count_dic[row[0]]
            elapsed_time_u_avg[num] = elapsed_time_u_sum_dic[row[0]] / answered_correctly_u_count_dic[row[0]]
            explanation_u_avg[num] = explanation_u_sum_dic[row[0]] / answered_correctly_u_count_dic[row[0]]
        else:
            answered_correctly_u_avg[num] = np.nan
            elapsed_time_u_avg[num] = np.nan
            explanation_u_avg[num] = np.nan


        answered_correctly_u_count_dic[row[0]] += 1
        elapsed_time_u_sum_dic[row[0]] += row[3]
        explanation_u_sum_dic[row[0]] += int(row[4])


    def update_time_stamp_user(self,num,row,
                                timestamp_u,timestamp_u_incorrect_dic,timestamp_u_incorrect_recency_dic, # dic
                                timestamp_u_recency_1,timestamp_u_recency_2,timestamp_u_recency_3        # df
                                ):
        if len(timestamp_u[row[0]]) == 0:
            timestamp_u_recency_1[num] = np.nan
            timestamp_u_recency_2[num] = np.nan
            timestamp_u_recency_3[num] = np.nan
        elif len(timestamp_u[row[0]]) == 1:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][0]
            timestamp_u_recency_2[num] = np.nan
            timestamp_u_recency_3[num] = np.nan
        elif len(timestamp_u[row[0]]) == 2:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][1]
            timestamp_u_recency_2[num] = row[5] - timestamp_u[row[0]][0]
            timestamp_u_recency_3[num] = np.nan
        elif len(timestamp_u[row[0]]) == 3:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][2]
            timestamp_u_recency_2[num] = row[5] - timestamp_u[row[0]][1]
            timestamp_u_recency_3[num] = row[5] - timestamp_u[row[0]][0]

        if len(timestamp_u_incorrect_dic[row[0]]) == 0:
            timestamp_u_incorrect_recency_dic[num] = np.nan
        else:
            timestamp_u_incorrect_recency_dic[num] = row[5] - timestamp_u_incorrect_dic[row[0]][0]


        if len(timestamp_u[row[0]]) == 3:
            timestamp_u[row[0]].pop(0)
            timestamp_u[row[0]].append(row[5])
        else:
            timestamp_u[row[0]].append(row[5])

    def questions_feats_update(self,num,row,
                                answered_correctly_q_count_dic,answered_correctly_q_sum_dic,elapsed_time_q_sum_dic,explanation_q_sum_dic, # dic
                                answered_correctly_q_avg,elapsed_time_q_avg,explanation_q_avg # df
                                ):
        if answered_correctly_q_count_dic[row[2]] != 0:
            answered_correctly_q_avg[num] = answered_correctly_q_sum_dic[row[2]] / answered_correctly_q_count_dic[row[2]]
            elapsed_time_q_avg[num] = elapsed_time_q_sum_dic[row[2]] / answered_correctly_q_count_dic[row[2]]
            explanation_q_avg[num] = explanation_q_sum_dic[row[2]] / answered_correctly_q_count_dic[row[2]]
        else:
            answered_correctly_q_avg[num] = np.nan
            elapsed_time_q_avg[num] = np.nan
            explanation_q_avg[num] = np.nan

        answered_correctly_q_count_dic[row[2]] += 1
        elapsed_time_q_sum_dic[row[2]] += row[3]
        explanation_q_sum_dic[row[2]] += int(row[4])

    def user_questions_feats_update(self,num,row,
                                    answered_correctly_uq_dic, # dic
                                    answered_correctly_uq_count # df
                                    ):

        answered_correctly_uq_count[num] = answered_correctly_uq_dic[row[0]][row[2]]
        answered_correctly_uq_dic[row[0]][row[2]] += 1

    def tags1_feats_update(self,num,row,
                        answered_correctly_tags1_count_dic,answered_correctly_tags1_sum_dic,elapsed_time_tags1_sum_dic,explanation_tags1_sum_dic, # dic
                        answered_correctly_tags1_avg,elapsed_time_tags1_avg,explanation_tags1_avg # df
                        ):
        if answered_correctly_tags1_count_dic[row[6]] != 0:
            answered_correctly_tags1_avg[num] = answered_correctly_tags1_sum_dic[row[6]] / answered_correctly_tags1_count_dic[row[6]]
            elapsed_time_tags1_avg[num] = elapsed_time_tags1_sum_dic[row[6]] / answered_correctly_tags1_count_dic[row[6]]
            explanation_tags1_avg[num] = explanation_tags1_sum_dic[row[6]] / answered_correctly_tags1_count_dic[row[6]]
        else:
            answered_correctly_tags1_avg[num] = np.nan
            elapsed_time_tags1_avg[num] = np.nan
            explanation_tags1_avg[num] = np.nan

        answered_correctly_tags1_count_dic[row[6]] += 1
        elapsed_time_tags1_sum_dic[row[6]] += row[3]
        explanation_tags1_sum_dic[row[6]] += int(row[4])


    def user_tags1_feats_update(self,num,row,
                                answered_correctly_ut_dic, # dic
                                answered_correctly_ut_count # df
                                ):

        answered_correctly_ut_count[num] = answered_correctly_ut_dic[row[0]][row[6]]
        answered_correctly_ut_dic[row[0]][row[6]] += 1


    def ans_update(self,num,row,
                   answered_correctly_u_sum_dic,timestamp_u_incorrect_dic,answered_correctly_q_sum_dic,answered_correctly_tags1_sum_dic,
                   update):

        # ------------------------------------------------------------------
        # Flag for training and inference
        if update:
            # ------------------------------------------------------------------
            # Client features updates
            answered_correctly_u_sum_dic[row[0]] += row[1]
            if row[1] == 0:
                if len(timestamp_u_incorrect_dic[row[0]]) == 1:
                    timestamp_u_incorrect_dic[row[0]].pop(0)
                    timestamp_u_incorrect_dic[row[0]].append(row[5])
                else:
                    timestamp_u_incorrect_dic[row[0]].append(row[5])

            # ------------------------------------------------------------------
            # Question features updates
            answered_correctly_q_sum_dic[row[2]] += row[1]
            # ------------------------------------------------------------------
            # Tags1 features updates
            answered_correctly_tags1_sum_dic[row[6]] += row[1]





    def add_past_feature(self,df, features_dicts,update = True):

        answered_correctly_u_count_dic = features_dicts['answered_correctly_user_count']
        answered_correctly_u_sum_dic = features_dicts['answered_correctly_user_sum']
        elapsed_time_u_sum_dic = features_dicts['elapsed_time_user_sum']
        explanation_u_sum_dic = features_dicts['explanation_user_sum']
        answered_correctly_q_count_dic = features_dicts['answered_correctly_q_count']
        answered_correctly_q_sum_dic = features_dicts['answered_correctly_q_sum']
        elapsed_time_q_sum_dic = features_dicts['elapsed_time_q_sum']
        explanation_q_sum_dic = features_dicts['explanation_q_sum']
        answered_correctly_uq_dic = features_dicts['answered_correctly_uq']
        timestamp_u = features_dicts['timestamp_u']
        timestamp_u_incorrect_dic = features_dicts['timestamp_u_incorrect']

        answered_correctly_tags1_count_dic = features_dicts['answered_correctly_tags1_count']
        answered_correctly_tags1_sum_dic = features_dicts['answered_correctly_tags1_sum']
        elapsed_time_tags1_sum_dic = features_dicts['answered_correctly_tags1_sum']
        explanation_tags1_sum_dic = features_dicts['explanation_tags1_sum_dic']

        answered_correctly_ut_dic = features_dicts['answered_correctly_ut']


        answered_correctly_u_avg = np.zeros(len(df), dtype = np.float32)
        elapsed_time_u_avg = np.zeros(len(df), dtype = np.float32)
        explanation_u_avg = np.zeros(len(df), dtype = np.float32)
        timestamp_u_recency_1 = np.zeros(len(df), dtype = np.float32)
        timestamp_u_recency_2 = np.zeros(len(df), dtype = np.float32)
        timestamp_u_recency_3 = np.zeros(len(df), dtype = np.float32)
        timestamp_u_incorrect_recency_dic = np.zeros(len(df), dtype = np.float32)
        # -----------------------------------------------------------------------
        # Question features
        answered_correctly_q_avg = np.zeros(len(df), dtype = np.float32)
        elapsed_time_q_avg = np.zeros(len(df), dtype = np.float32)
        explanation_q_avg = np.zeros(len(df), dtype = np.float32)
        # -----------------------------------------------------------------------
        # User Question
        answered_correctly_uq_count = np.zeros(len(df), dtype = np.int32)
        # -----------------------------------------------------------------------
        # Tags1 features
        answered_correctly_tags1_avg = np.zeros(len(df), dtype = np.float32)
        elapsed_time_tags1_avg = np.zeros(len(df), dtype = np.float32)
        explanation_tags1_avg = np.zeros(len(df), dtype = np.float32)
        # -----------------------------------------------------------------------
        # User Tags1
        answered_correctly_ut_count = np.zeros(len(df), dtype = np.int32)


        for num, row in enumerate(tqdm(df[['user_id', 'answered_correctly', 'content_id',
                                        'prior_question_elapsed_time', 'prior_question_had_explanation', 'timestamp','tags1']].values)):

            # User
            # ------------------------------------------------------------------
            self.user_feats_update(num,row,
                                   answered_correctly_u_count_dic,answered_correctly_u_sum_dic,elapsed_time_u_sum_dic,explanation_u_sum_dic, # dic
                                   answered_correctly_u_avg,elapsed_time_u_avg,explanation_u_avg # df
                                   )

            # ------------------------------------------------------------------
            # Time
            self.update_time_stamp_user(num,row,
                                        timestamp_u,timestamp_u_incorrect_dic,timestamp_u_incorrect_recency_dic,
                                        timestamp_u_recency_1,timestamp_u_recency_2,timestamp_u_recency_3
                                        )

            # ------------------------------------------------------------------
            # Question
            self.questions_feats_update(num,row,
                                        answered_correctly_q_count_dic,answered_correctly_q_sum_dic,elapsed_time_q_sum_dic,explanation_q_sum_dic, # dic
                                        answered_correctly_q_avg,elapsed_time_q_avg,explanation_q_avg # df
                                        )

            # ------------------------------------------------------------------
            # User Question
            self.user_questions_feats_update(num,row,
                                            answered_correctly_uq_dic, # dic
                                            answered_correctly_uq_count # df
                                            )

            # ------------------------------------------------------------------
            # Tags1
            self.tags1_feats_update(num,row,
                                    answered_correctly_tags1_count_dic,answered_correctly_tags1_sum_dic,elapsed_time_tags1_sum_dic,explanation_tags1_sum_dic, # dic
                                    answered_correctly_tags1_avg,elapsed_time_tags1_avg,explanation_tags1_avg # df
                                    )

            # ------------------------------------------------------------------
            # Tags1 Question
            self.user_tags1_feats_update(num,row,
                                        answered_correctly_ut_dic, # dic
                                        answered_correctly_ut_count # df
                                        )

            # ------------------------------------------------------------------
            # 予測時にはupdateしない。
            self.ans_update(num,row,
                            answered_correctly_u_sum_dic,timestamp_u_incorrect_dic,answered_correctly_q_sum_dic,answered_correctly_tags1_sum_dic,
                            update)


        user_df = pd.DataFrame({'answered_correctly_user_avg': answered_correctly_u_avg, 'elapsed_time_user_avg': elapsed_time_u_avg, 'explanation_user_avg': explanation_u_avg,
                                'answered_correctly_q_avg': answered_correctly_q_avg, 'elapsed_time_q_avg': elapsed_time_q_avg, 'explanation_q_avg': explanation_q_avg,
                                'answered_correctly_uq_count': answered_correctly_uq_count, 'timestamp_u_recency_1': timestamp_u_recency_1, 'timestamp_u_recency_2': timestamp_u_recency_2,
                                'timestamp_u_recency_3': timestamp_u_recency_3, 'timestamp_u_incorrect_recency': timestamp_u_incorrect_recency_dic,
                                'answered_correctly_tags1_avg' : answered_correctly_tags1_avg,'elapsed_time_tags1_avg': elapsed_time_tags1_avg,'explanation_tags1_avg':explanation_tags1_avg,
                                'answered_correctly_ut_count' : answered_correctly_ut_count})

        features_dicts = {
                            'answered_correctly_user_count': answered_correctly_u_count_dic,
                            'answered_correctly_user_sum': answered_correctly_u_sum_dic,
                            'elapsed_time_user_sum': elapsed_time_u_sum_dic,
                            'explanation_user_sum': explanation_u_sum_dic,
                            'answered_correctly_q_count': answered_correctly_q_count_dic,
                            'answered_correctly_q_sum': answered_correctly_q_sum_dic,
                            'elapsed_time_q_sum': elapsed_time_q_sum_dic,
                            'explanation_q_sum': explanation_q_sum_dic,
                            'answered_correctly_uq': answered_correctly_uq_dic,
                            'timestamp_u': timestamp_u,
                            'timestamp_u_incorrect': timestamp_u_incorrect_dic,
                            'answered_correctly_tags1_count' : answered_correctly_tags1_count_dic,
                            'answered_correctly_tags1_sum' : answered_correctly_tags1_sum_dic,
                            'elapsed_time_tags1_sum' : elapsed_time_tags1_sum_dic,
                            'explanation_tags1_sum_dic' : explanation_tags1_sum_dic,
                            'answered_correctly_ut' : answered_correctly_ut_dic
                        }



        df = pd.concat([df, user_df], axis = 1)
        return df

    def create_features(self):
        create_feats = ['answered_correctly_user_avg','elapsed_time_user_avg','explanation_user_avg',\
                        'answered_correctly_q_avg','elapsed_time_q_avg','explanation_q_avg',\
                        'answered_correctly_uq_count','timestamp_u_recency_1','timestamp_u_recency_2',\
                        'timestamp_u_recency_3','timestamp_u_incorrect_recency',\
                        'answered_correctly_tags1_avg','elapsed_time_tags1_avg','explanation_tags1_avg',
                        'answered_correctly_ut_count']

        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')

        answered_correctly_u_count_dic = defaultdict(int)
        answered_correctly_u_sum_dic = defaultdict(int)
        elapsed_time_u_sum_dic = defaultdict(int)
        explanation_u_sum_dic = defaultdict(int)
        timestamp_u = defaultdict(list)
        timestamp_u_incorrect_dic = defaultdict(list)

        # Question dictionaries
        answered_correctly_q_count_dic = defaultdict(int)
        answered_correctly_q_sum_dic = defaultdict(int)
        elapsed_time_q_sum_dic = defaultdict(int)
        explanation_q_sum_dic = defaultdict(int)

        # tags dict
        answered_correctly_tags1_count_dic = defaultdict(int)
        answered_correctly_tags1_sum_dic = defaultdict(int)
        elapsed_time_tags1_sum_dic = defaultdict(int)
        explanation_tags1_sum_dic = defaultdict(int)

        # User Question dictionary
        answered_correctly_uq_dic = defaultdict(lambda: defaultdict(int))

        # User Tags1 dictionary
        answered_correctly_ut_dic = defaultdict(lambda: defaultdict(int))

        features_dicts = {
                            'answered_correctly_user_count': answered_correctly_u_count_dic,
                            'answered_correctly_user_sum': answered_correctly_u_sum_dic,
                            'elapsed_time_user_sum': elapsed_time_u_sum_dic,
                            'explanation_user_sum': explanation_u_sum_dic,
                            'answered_correctly_q_count': answered_correctly_q_count_dic,
                            'answered_correctly_q_sum': answered_correctly_q_sum_dic,
                            'elapsed_time_q_sum': elapsed_time_q_sum_dic,
                            'explanation_q_sum': explanation_q_sum_dic,
                            'answered_correctly_uq': answered_correctly_uq_dic,
                            'timestamp_u': timestamp_u,
                            'timestamp_u_incorrect': timestamp_u_incorrect_dic,
                            'answered_correctly_tags1_count' : answered_correctly_tags1_count_dic,
                            'answered_correctly_tags1_sum' : answered_correctly_tags1_sum_dic,
                            'elapsed_time_tags1_sum' : elapsed_time_tags1_sum_dic,
                            'explanation_tags1_sum_dic' : explanation_tags1_sum_dic,
                            'answered_correctly_ut' : answered_correctly_ut_dic
                        }




        self.train = self.add_past_feature(self.train, features_dicts)

        self.valid = self.add_past_feature(self.valid, features_dicts)

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]

        # if not MINI_DATA:
        #     with open('./features/kernel_data/loop_feats.pkl','wb') as f:
        #         pickle.dump(features_dicts,f)


class CROSS_FEATURES(Feature):

    def create_features(self):
        create_feats = ['answered_correctly_avg_user_q']
        FEATURES_LIST = ['BASE','LOOP']
        self.train = data_util.load_features(FEATURES_LIST,path=f'{Feature.dir}',train_valid='train')
        self.valid = data_util.load_features(FEATURES_LIST,path=f'{Feature.dir}',train_valid='valid')

        self.train['answered_correctly_avg_user_q'] = self.train['answered_correctly_user_avg'] * self.train['answered_correctly_q_avg']
        self.valid['answered_correctly_avg_user_q'] = self.valid['answered_correctly_user_avg'] * self.valid['answered_correctly_q_avg']




        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]



if __name__ == "__main__":
    args = get_arguments()
    generate_features(globals(),args.force)