import pandas as pd
import numpy as np
import gc
import os
from base_create_features import Feature, get_arguments, generate_features
from pathlib import Path
Feature.dir = 'features/data'

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


# def target_encoding(data,col):
#     group_cols =
#     data[''] = train.groupby([TARGET,]).mean()



class BASE(Feature):

    def create_features(self):
        self.train = pd.read_feather('./data/input/train.feather')
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


        self.train = self.train[self.train['content_type_id'] == 0].reset_index(drop=True)


class USER_ID(Feature):

    def create_features(self):
        create_cols = ['past_correctly_sum','past_not_correctly_sum','past_correctly_mean']
        self.train = pd.read_feather('./features/data/BASE_train.feather')
        self.train = self.train[['user_id','answered_correctly']]
        encoded = (self.train.groupby('user_id')['answered_correctly']
                   .expanding().agg(['mean','sum','count']).groupby('user_id').shift(1))
        encoded.columns = ['past_correctly_mean','past_correctly_sum','count']
        encoded['past_not_correctly_sum'] = encoded['count'] - encoded['past_correctly_sum']

        encoded = encoded.reset_index(drop=True)
        self.train = encoded[create_cols]




class PART(Feature):

    def create_features(self):
        # 少しリークしているが、これくらいならOK判断
        create_cols = ['user_part_mean','user_part_sum','part_ans_mean']
        self.train = pd.read_feather('./features/data/BASE_train.feather')
        self.train = self.train[['user_id','answered_correctly','part']]
        part_ans_mean = self.train[['part','answered_correctly']].groupby('part').mean()
        part_ans_mean.columns = ['part_ans_mean']
        self.train = pd.merge(self.train,part_ans_mean,on='part',how='left')
        del part_ans_mean
        gc.collect()

        encoded = (self.train.groupby(['user_id','part'])['answered_correctly']
                   .expanding().agg(['mean','sum']).groupby('user_id').shift(1))
        encoded = encoded.reset_index().sort_values('level_2').reset_index(drop=True)
        encoded.columns = ['user_id','part','level_2','user_part_mean','user_part_sum']

        self.train = pd.concat([self.train,encoded[['user_part_mean', 'user_part_sum']]],axis=1)
        del encoded
        gc.collect()

        self.train = self.train[create_cols]


class CONTENT(Feature):

    def create_features(self):
        # 少しリークしているが、これくらいならOK判断
        create_cols = ['content_id_ans_mean','content_id_num']
        # TODO ,'content_id_user_mean', 'content_id_user_sum'
        self.train = pd.read_feather('./features/data/BASE_train.feather')
        self.train = self.train[['user_id','answered_correctly','content_id']]
        content_ans_mean = self.train[['content_id','answered_correctly']].groupby('content_id').mean()
        content_ans_mean.columns = ['content_id_ans_mean']
        content_sum = pd.DataFrame(self.train['content_id'].value_counts().sort_values(ascending=False)).reset_index()
        content_sum.columns = ['content_id','content_id_num']

        self.train = pd.merge(self.train,content_ans_mean,on='content_id',how='left')
        self.train = pd.merge(self.train,content_sum,on='content_id',how='left')
        del content_ans_mean,content_sum
        gc.collect()

        # メモリ不足なので他の方法を考える
        # encoded = (self.train.groupby(['user_id','content_id'])['answered_correctly']
        #         .expanding().agg(['mean','sum']).groupby('content_id').shift(1))
        # encoded = encoded.reset_index().sort_values('level_2').reset_index(drop=True).fillna(0)
        # encoded.columns = ['user_id', 'content_id', 'level_2', 'content_id_user_mean', 'content_id_user_sum']
        # self.train = pd.concat([self.train,encoded[['content_id_user_mean', 'content_id_user_sum']]],axis=1)
        # del encoded
        # gc.collect()

        self.train = self.train[create_cols]


class TARGET_ENCODING(Feature):

    def target_encoding(self,data,col_list):
        encoding_cols = col.copy()
        encoding_cols += TARGET

        feature_name =
        data[]


    def create_features(self):

        self.train = pd.read_feather('./features/data/BASE_train.feather')
        create_cols = ['tag1_mean','tag2_mean','bundle_id_mean','bundle_id_tag1_mean','task_container_id_mean']

        cols = ['tag1','tag2','bundle_id',['bundle_id','tag1'],'task_container_id']


        for col in cols:
            self.target_encoding(self.train,col)




# class TIME_STAMP(self):

#     def create_features(self):



if __name__ == "__main__":
    args = get_arguments()
    generate_features(globals(),args.force)