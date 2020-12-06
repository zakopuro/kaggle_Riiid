import pandas as pd
import numpy as np
import gc
import os
from tqdm import tqdm
from collections import defaultdict
from base_create_features import Feature, get_arguments, generate_features
from pathlib import Path
Feature.dir = 'features/mini_data'

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
        train_index = pd.read_feather('./data/train_valid/cv37_train.feather')
        valid_index = pd.read_feather('./data/train_valid/cv37_valid.feather')

        qs = pd.read_csv('./data/input/questions.csv')
        lc = pd.read_csv('./data/input/lectures_new.csv')
        tag = qs["tags"].str.split(" ",expand = True)
        tag.columns = ['tags1','tags2','tags3','tags4','tags5','tags6']
        qs = pd.concat([qs,tag],axis=1)
        lc['l_type_of'] = _label_encoder(lc['type_of'])
        qs = qs.rename(columns={'question_id':'content_id'})
        lc = lc.rename(columns={'lecture_id':'content_id'})
        qs_lc = pd.concat([qs,lc])

        self.train.loc[self.train['prior_question_had_explanation'] == False , 'prior_question_had_explanation'] = 0
        self.train.loc[self.train['prior_question_had_explanation'] == True , 'prior_question_had_explanation'] = 1
        self.train['prior_question_had_explanation'] = self.train['prior_question_had_explanation'].astype(float)

        self.train = pd.merge(self.train,qs_lc,on='content_id',how='left')
        for i in range(1,7):
            self.train[f'tags{i}'] = self.train[f'tags{i}'].astype(float)

        self.valid = self.train[self.train['row_id'].isin(valid_index['row_id'])]
        self.train = self.train[self.train['row_id'].isin(train_index['row_id'])]


        self.train = self.train[self.train['content_type_id'] == 0].reset_index(drop=True)
        self.valid = self.valid[self.valid['content_type_id'] == 0].reset_index(drop=True)


class USER_ID(Feature):

    def user_past_features(self,df,answered_correctly_sum_u_dict, count_u_dict):
        acsu = np.zeros(len(df), dtype=np.int32)
        cu = np.zeros(len(df), dtype=np.int32)
        for cnt,row in enumerate(tqdm(df[['user_id',TARGET]].values)):
            acsu[cnt] = answered_correctly_sum_u_dict[row[0]]
            cu[cnt] = count_u_dict[row[0]]
            answered_correctly_sum_u_dict[row[0]] += row[1]
            count_u_dict[row[0]] += 1
        user_feats_df = pd.DataFrame({'answered_correctly_sum_user':acsu, 'count_user':cu})
        user_feats_df['answered_correctly_avg_user'] = user_feats_df['answered_correctly_sum_user'] / user_feats_df['count_user']
        df = pd.concat([df, user_feats_df], axis=1)
        return df

    def create_features(self):
        create_feats = ['answered_correctly_avg_user','answered_correctly_sum_user','count_user']
        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')
        answered_correctly_sum_u_dict = defaultdict(int)
        count_u_dict = defaultdict(int)

        self.train = self.user_past_features(self.train, answered_correctly_sum_u_dict, count_u_dict)
        self.valid = self.user_past_features(self.valid, answered_correctly_sum_u_dict, count_u_dict)

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]




class PART(Feature):

    def create_features(self):
        # 少しリークしているが、これくらいならOK判断
        create_feats = ['answered_correctly_avg_part']
        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')
        self.train = self.train[['part',TARGET]]
        self.valid = self.valid[['part',TARGET]]

        part_mean = self.train[['part',TARGET]].groupby(['part']).agg(['mean'])
        part_mean.columns = ['answered_correctly_avg_part']
        self.train = pd.merge(self.train, part_mean, on=['part'], how="left")
        self.valid = pd.merge(self.valid, part_mean, on=['part'], how="left")

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]


class USER_PART(Feature):

    def user_part_past_features(self,df,answered_correctly_sum_u_p_dict, count_u_p_dict):
        acsu = np.zeros(len(df), dtype=np.int32)
        cu = np.zeros(len(df), dtype=np.int32)
        for cnt,row in enumerate(tqdm(df[['user_id_part',TARGET]].values)):
            acsu[cnt] = answered_correctly_sum_u_p_dict[row[0]]
            cu[cnt] = count_u_p_dict[row[0]]
            answered_correctly_sum_u_p_dict[row[0]] += row[1]
            count_u_p_dict[row[0]] += 1
        user_feats_df = pd.DataFrame({'answered_correctly_sum_user_part':acsu, 'count_user_part':cu})
        user_feats_df['answered_correctly_avg_user_part'] = user_feats_df['answered_correctly_sum_user_part'] / user_feats_df['count_user_part']
        df = pd.concat([df, user_feats_df], axis=1)
        return df


    def create_features(self):
        create_feats = ['answered_correctly_avg_user_part','answered_correctly_sum_user_part','count_user_part']

        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')
        self.train['user_id_part'] = self.train['user_id'].astype(str) + '-' + self.train['part'].astype(str)
        self.valid['user_id_part'] = self.valid['user_id'].astype(str) + '-' + self.valid['part'].astype(str)
        answered_correctly_sum_u_p_dict = defaultdict(int)
        count_u_p_dict = defaultdict(int)

        self.train = self.user_part_past_features(self.train, answered_correctly_sum_u_p_dict, count_u_p_dict)
        self.valid = self.user_part_past_features(self.valid, answered_correctly_sum_u_p_dict, count_u_p_dict)

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]



class CONTENT(Feature):

    def create_features(self):
        create_feats = ['answered_correctly_avg_content','answered_correctly_sum_content','content_num']
        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')

        content_ans_av_sum = self.train[['content_id',TARGET]].groupby('content_id')[TARGET].agg(['mean','sum'])
        content_ans_av_sum.columns = ['answered_correctly_avg_content','answered_correctly_sum_content']

        self.train = pd.merge(self.train,content_ans_av_sum,on='content_id',how='left')
        self.valid = pd.merge(self.valid,content_ans_av_sum,on='content_id',how='left')

        content_num = pd.DataFrame(self.train['content_id'].value_counts().sort_values(ascending=False)).reset_index()
        content_num.columns = ['content_id','content_num']

        self.train = pd.merge(self.train,content_num,on='content_id',how='left')
        self.valid = pd.merge(self.valid,content_num,on='content_id',how='left')

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]


class USER_CONTENT(Feature):
    def user_content_past_features(self,df,answered_correctly_sum_u_c_dict, count_u_c_dict):
        acsu = np.zeros(len(df), dtype=np.int32)
        cu = np.zeros(len(df), dtype=np.int32)
        for cnt,row in enumerate(tqdm(df[['user_id_content',TARGET]].values)):
            acsu[cnt] = answered_correctly_sum_u_c_dict[row[0]]
            cu[cnt] = count_u_c_dict[row[0]]
            answered_correctly_sum_u_c_dict[row[0]] += row[1]
            count_u_c_dict[row[0]] += 1
        user_feats_df = pd.DataFrame({'answered_correctly_sum_user_content':acsu, 'count_user_content':cu})
        user_feats_df['answered_correctly_avg_user_content'] = user_feats_df['answered_correctly_sum_user_content'] / user_feats_df['count_user_content']
        df = pd.concat([df, user_feats_df], axis=1)
        return df


    def create_features(self):
        create_feats = ['answered_correctly_avg_user_content','answered_correctly_sum_user_content','count_user_content']

        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')
        self.train['user_id_content'] = self.train['user_id'].astype(str) + '-' + self.train['content_id'].astype(str)
        self.valid['user_id_content'] = self.valid['user_id'].astype(str) + '-' + self.valid['content_id'].astype(str)
        answered_correctly_sum_u_c_dict = defaultdict(int)
        count_u_c_dict = defaultdict(int)

        self.train = self.user_content_past_features(self.train, answered_correctly_sum_u_c_dict, count_u_c_dict)
        self.valid = self.user_content_past_features(self.valid, answered_correctly_sum_u_c_dict, count_u_c_dict)

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]




class TAGS(Feature):

    def create_features(self):
        create_feats = ['answered_correctly_avg_tags1']
        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')

        tag1_mean = self.train[['tags1',TARGET]].groupby(['tags1']).agg(['mean'])
        tag1_mean.columns = ['answered_correctly_avg_tags1']

        self.train = pd.merge(self.train, tag1_mean, on=['tags1'], how="left")
        self.valid = pd.merge(self.valid, tag1_mean, on=['tags1'], how="left")

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]




if __name__ == "__main__":
    args = get_arguments()
    generate_features(globals(),args.force)