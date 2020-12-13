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

MINI_DATA = True
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

        self.test = self.train.copy()
        self.valid = self.train[self.train['row_id'].isin(valid_index['row_id'])]
        self.train = self.train[self.train['row_id'].isin(train_index['row_id'])]


        self.train = self.train[self.train['content_type_id'] == 0].reset_index(drop=True)
        self.valid = self.valid[self.valid['content_type_id'] == 0].reset_index(drop=True)
        # self.test = self.test[self.test['content_type_id'] == 0].reset_index(drop=True)

        prior_question_elapsed_time_mean = self.train.prior_question_elapsed_time.dropna().values.mean()
        self.train['prior_question_elapsed_time_mean'] = self.train.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)
        self.valid['prior_question_elapsed_time_mean'] = self.valid.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)


        self.test = pd.DataFrame()



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

        if not MINI_DATA:
            with open(f'./{Feature.kernel_dir}/user_dict_sum.pkl','wb') as f:
                pickle.dump(answered_correctly_sum_u_dict,f)
            with open(f'./{Feature.kernel_dir}/user_dict_count.pkl','wb') as f:
                pickle.dump(count_u_dict,f)

class CONTENT(Feature):

    def create_features(self):
        create_feats = ['answered_correctly_avg_content','answered_correctly_sum_content','answered_correctly_std_content','content_num']
        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')

        content_ans_av_sum = self.train[['content_id',TARGET]].groupby('content_id')[TARGET].agg(['mean','sum','std'])
        content_ans_av_sum.columns = ['answered_correctly_avg_content','answered_correctly_sum_content','answered_correctly_std_content']

        self.train = pd.merge(self.train,content_ans_av_sum,on='content_id',how='left')
        self.valid = pd.merge(self.valid,content_ans_av_sum,on='content_id',how='left')

        content_num = pd.DataFrame(self.train['content_id'].value_counts().sort_values(ascending=False)).reset_index()
        content_num.columns = ['content_id','content_num']

        self.train = pd.merge(self.train,content_num,on='content_id',how='left')
        self.valid = pd.merge(self.valid,content_num,on='content_id',how='left')


        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]

        if not MINI_DATA:
            self.test = pd.read_feather(f'./{Feature.kernel_dir}/BASE_test.feather')

            content_ans_av_sum = self.test[['content_id',TARGET]].groupby('content_id')[TARGET].agg(['mean','sum','std'])
            content_ans_av_sum.columns = ['answered_correctly_avg_content','answered_correctly_sum_content','answered_correctly_std_content']

            content_num = pd.DataFrame(self.test['content_id'].value_counts().sort_values(ascending=False)).reset_index()
            content_num.columns = ['content_id','content_num']

            content_ans_av_sum = content_ans_av_sum.reset_index()
            content_num = content_num.reset_index()
            content_feats = pd.merge(content_ans_av_sum,content_num,on='content_id',how='left')

            self.test = content_feats

class CONTENT_CUT(Feature):

    def create_features(self):
        create_feats = ['answered_correctly_avg_content_cut','answered_correctly_sum_content_cut','answered_correctly_std_content_cut','content_cut_num','content_id_cut']
        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')

        self.train['content_id_cut'] = self.train['content_id']/100
        self.valid['content_id_cut'] = self.valid['content_id']/100
        self.train['content_id_cut'] = self.train['content_id_cut'].astype(int)
        self.valid['content_id_cut'] = self.valid['content_id_cut'].astype(int)

        content_ans_av_sum = self.train[['content_id_cut',TARGET]].groupby('content_id_cut')[TARGET].agg(['mean','sum','std'])
        content_ans_av_sum.columns = ['answered_correctly_avg_content_cut','answered_correctly_sum_content_cut','answered_correctly_std_content_cut']

        self.train = pd.merge(self.train,content_ans_av_sum,on='content_id_cut',how='left')
        self.valid = pd.merge(self.valid,content_ans_av_sum,on='content_id_cut',how='left')

        content_num = pd.DataFrame(self.train['content_id_cut'].value_counts().sort_values(ascending=False)).reset_index()
        content_num.columns = ['content_id_cut','content_cut_num']

        self.train = pd.merge(self.train,content_num,on='content_id_cut',how='left')
        self.valid = pd.merge(self.valid,content_num,on='content_id_cut',how='left')


        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]

        if not MINI_DATA:
            self.test = pd.read_feather(f'./{Feature.kernel_dir}/BASE_test.feather')
            self.test['content_id_cut'] = self.test['content_id']/100
            self.test['content_id_cut'] = self.test['content_id_cut'].astype(int)

            content_ans_av_sum = self.test[['content_id_cut',TARGET]].groupby('content_id_cut')[TARGET].agg(['mean','sum','std'])
            content_ans_av_sum.columns = ['answered_correctly_avg_content_cut','answered_correctly_sum_content_cut','answered_correctly_std_content_cut']

            content_num = pd.DataFrame(self.test['content_id_cut'].value_counts().sort_values(ascending=False)).reset_index()
            content_num.columns = ['content_id_cut','content_cut_num']

            content_ans_av_sum = content_ans_av_sum.reset_index()
            content_num = content_num.reset_index()
            content_feats = pd.merge(content_ans_av_sum,content_num,on='content_id_cut',how='left')

            self.test = content_feats

class BUNDLE(Feature):

    def create_features(self):
        create_feats = ['answered_correctly_avg_bundle','answered_correctly_sum_bundle','answered_correctly_std_bundle','bundle_num']
        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')

        bundle_ans_av_sum = self.train[['bundle_id',TARGET]].groupby('bundle_id')[TARGET].agg(['mean','sum','std'])
        bundle_ans_av_sum.columns = ['answered_correctly_avg_bundle','answered_correctly_sum_bundle','answered_correctly_std_bundle']

        self.train = pd.merge(self.train,bundle_ans_av_sum,on='bundle_id',how='left')
        self.valid = pd.merge(self.valid,bundle_ans_av_sum,on='bundle_id',how='left')

        bundle_num = pd.DataFrame(self.train['bundle_id'].value_counts().sort_values(ascending=False)).reset_index()
        bundle_num.columns = ['bundle_id','bundle_num']

        self.train = pd.merge(self.train,bundle_num,on='bundle_id',how='left')
        self.valid = pd.merge(self.valid,bundle_num,on='bundle_id',how='left')


        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]

        if not MINI_DATA:
            self.test = pd.read_feather(f'./{Feature.kernel_dir}/BASE_test.feather')

            bundle_ans_av_sum = self.test[['bundle_id',TARGET]].groupby('bundle_id')[TARGET].agg(['mean','sum','std'])
            bundle_ans_av_sum.columns = ['answered_correctly_avg_bundle','answered_correctly_sum_bundle','answered_correctly_std_bundle']

            bundle_num = pd.DataFrame(self.test['bundle_id'].value_counts().sort_values(ascending=False)).reset_index()
            bundle_num.columns = ['bundle_id','bundle_num']

            bundle_ans_av_sum = bundle_ans_av_sum.reset_index()
            bundle_num = bundle_num.reset_index()
            bundle_feats = pd.merge(bundle_ans_av_sum,bundle_num,on='bundle_id',how='left')

            self.test = bundle_feats

class BUNDLE_CUT(Feature):

    def create_features(self):
        create_feats = ['answered_correctly_avg_bundle_cut','answered_correctly_sum_bundle_cut','answered_correctly_std_bundle_cut','bundle_cut_num','bundle_id_cut']
        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')

        self.train['bundle_id_cut'] = self.train['bundle_id']/100
        self.valid['bundle_id_cut'] = self.valid['bundle_id']/100
        self.train['bundle_id_cut'] = self.train['bundle_id_cut'].astype(int)
        self.valid['bundle_id_cut'] = self.valid['bundle_id_cut'].astype(int)


        bundle_ans_av_sum = self.train[['bundle_id_cut',TARGET]].groupby('bundle_id_cut')[TARGET].agg(['mean','sum','std'])
        bundle_ans_av_sum.columns = ['answered_correctly_avg_bundle_cut','answered_correctly_sum_bundle_cut','answered_correctly_std_bundle_cut']

        self.train = pd.merge(self.train,bundle_ans_av_sum,on='bundle_id_cut',how='left')
        self.valid = pd.merge(self.valid,bundle_ans_av_sum,on='bundle_id_cut',how='left')

        bundle_num = pd.DataFrame(self.train['bundle_id_cut'].value_counts().sort_values(ascending=False)).reset_index()
        bundle_num.columns = ['bundle_id_cut','bundle_cut_num']

        self.train = pd.merge(self.train,bundle_num,on='bundle_id_cut',how='left')
        self.valid = pd.merge(self.valid,bundle_num,on='bundle_id_cut',how='left')


        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]

        if not MINI_DATA:
            self.test = pd.read_feather(f'./{Feature.kernel_dir}/BASE_test.feather')
            self.test['bundle_id_cut'] = self.test['bundle_id']/100
            self.test['bundle_id_cut'] = self.test['bundle_id_cut'].astype(int)

            bundle_ans_av_sum = self.test[['bundle_id_cut',TARGET]].groupby('bundle_id_cut')[TARGET].agg(['mean','sum','std'])
            bundle_ans_av_sum.columns = ['answered_correctly_avg_bundle_cut','answered_correctly_sum_bundle_cut','answered_correctly_std_bundle_cut']

            bundle_num = pd.DataFrame(self.test['bundle_id_cut'].value_counts().sort_values(ascending=False)).reset_index()
            bundle_num.columns = ['bundle_id_cut','bundle_cut_num']

            bundle_ans_av_sum = bundle_ans_av_sum.reset_index()
            bundle_num = bundle_num.reset_index()
            bundle_feats = pd.merge(bundle_ans_av_sum,bundle_num,on='bundle_id_cut',how='left')

            self.test = bundle_feats


class TASK_CONTAINER(Feature):

    def create_features(self):
        create_feats = ['answered_correctly_avg_task_container','answered_correctly_sum_task_container','answered_correctly_std_task_container']
        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')

        task_ans_av_sum = self.train[['task_container_id',TARGET]].groupby('task_container_id')[TARGET].agg(['mean','sum','std'])
        task_ans_av_sum.columns = ['answered_correctly_avg_task_container','answered_correctly_sum_task_container','answered_correctly_std_task_container']

        self.train = pd.merge(self.train,task_ans_av_sum,on='task_container_id',how='left')
        self.valid = pd.merge(self.valid,task_ans_av_sum,on='task_container_id',how='left')

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]

        if not MINI_DATA:
            self.test = pd.read_feather(f'./{Feature.kernel_dir}/BASE_test.feather')

            task_ans_av_sum = self.test[['task_container_id',TARGET]].groupby('task_container_id')[TARGET].agg(['mean','sum','std'])
            task_ans_av_sum.columns = ['answered_correctly_avg_task_container','answered_correctly_sum_task_container','answered_correctly_std_task_container']
            task_ans_av_sum = task_ans_av_sum.reset_index()
            self.test = task_ans_av_sum

class PART(Feature):
    def create_features(self):
        create_feats = ['answered_correctly_avg_part','answered_correctly_sum_part','answered_correctly_std_part']
        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')

        part_ans_av_sum = self.train[['part',TARGET]].groupby('part')[TARGET].agg(['mean','sum','std'])
        part_ans_av_sum.columns = ['answered_correctly_avg_part','answered_correctly_sum_part','answered_correctly_std_part']

        self.train = pd.merge(self.train,part_ans_av_sum,on='part',how='left')
        self.valid = pd.merge(self.valid,part_ans_av_sum,on='part',how='left')

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]

        if not MINI_DATA:
            self.test = pd.read_feather(f'./{Feature.kernel_dir}/BASE_test.feather')

            part_ans_av_sum = self.test[['part',TARGET]].groupby('part')[TARGET].agg(['mean','sum','std'])
            part_ans_av_sum.columns = ['answered_correctly_avg_part','answered_correctly_sum_part','answered_correctly_std_part']
            part_ans_av_sum = part_ans_av_sum.reset_index()
            self.test = part_ans_av_sum

class TAGS1(Feature):
    def create_features(self):
        create_feats = ['answered_correctly_avg_tags1','answered_correctly_sum_tags1','answered_correctly_std_tags1']
        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')

        tags1_ans_av_sum = self.train[['tags1',TARGET]].groupby('tags1')[TARGET].agg(['mean','sum','std'])
        tags1_ans_av_sum.columns = ['answered_correctly_avg_tags1','answered_correctly_sum_tags1','answered_correctly_std_tags1']

        self.train = pd.merge(self.train,tags1_ans_av_sum,on='tags1',how='left')
        self.valid = pd.merge(self.valid,tags1_ans_av_sum,on='tags1',how='left')

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]

        if not MINI_DATA:
            self.test = pd.read_feather(f'./{Feature.kernel_dir}/BASE_test.feather')

            tags1_ans_av_sum = self.test[['tags1',TARGET]].groupby('tags1')[TARGET].agg(['mean','sum','std'])
            tags1_ans_av_sum.columns = ['answered_correctly_avg_tags1','answered_correctly_sum_tags1','answered_correctly_std_tags1']
            tags1_ans_av_sum = tags1_ans_av_sum.reset_index()
            self.test = tags1_ans_av_sum

class TIME(Feature):

    def create_features(self):
        create_feats = ['lag_time','user_lag_time_mean','days_elapsed','lag_days','paid_user','lag_hours',\
                        'prior_question_elapsed_time_mean']
        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')

        self.train['lag_time'] = self.train.groupby('user_id')['timestamp'].shift()
        self.valid['lag_time'] = self.valid.groupby('user_id')['timestamp'].shift()
        self.train['lag_time'] = self.train['timestamp'] - self.train['lag_time']
        self.valid['lag_time'] = self.valid['timestamp'] - self.valid['lag_time']

        lag_time_mean = self.train.groupby('user_id')['lag_time'].agg(['mean'])
        self.train['user_lag_time_mean'] = self.train['user_id'].map(lag_time_mean['mean'])
        self.valid['user_lag_time_mean'] = self.valid['user_id'].map(lag_time_mean['mean'])

        self.train['days_elapsed'] = self.train['timestamp']/(1000*3600*24)
        self.valid['days_elapsed'] = self.valid['timestamp']/(1000*3600*24)
        self.train['days_elapsed'] = self.train['days_elapsed'].astype(int)
        self.valid['days_elapsed'] = self.valid['days_elapsed'].astype(int)

        self.train['hour_elapsed'] = self.train['timestamp']/(1000*3600)
        self.valid['hour_elapsed'] = self.valid['timestamp']/(1000*3600)
        self.train['hour_elapsed'] = self.train['hour_elapsed'].astype(int)
        self.valid['hour_elapsed'] = self.valid['hour_elapsed'].astype(int)


        self.train['lag_days'] = self.train.groupby('user_id')['days_elapsed'].shift()
        self.valid['lag_days'] = self.valid.groupby('user_id')['days_elapsed'].shift()
        self.train['lag_days'] = self.train['days_elapsed'] - self.train['lag_days']
        self.valid['lag_days'] = self.valid['days_elapsed'] - self.valid['lag_days']


        self.train['lag_hours'] = self.train.groupby('user_id')['hour_elapsed'].shift()
        self.valid['lag_hours'] = self.valid.groupby('user_id')['hour_elapsed'].shift()
        self.train['lag_hours'] = self.train['hour_elapsed'] - self.train['lag_hours']
        self.valid['lag_hours'] = self.valid['hour_elapsed'] - self.valid['lag_hours']



        # 有料会員/無料会員
        # 一日20問までなので20問より多く解いているユーザーは有料会員の可能性が高い。
        df = pd.concat([self.train,self.valid]).reset_index(drop=True)
        df['count'] = 1
        user_day_count = df[df['days_elapsed'] >= 1].groupby(['user_id','days_elapsed'])['count'].agg(['count'])
        user_day_count = user_day_count.reset_index()
        # 50にしておく
        paid_user = user_day_count[user_day_count['count'] > 50]['user_id'].unique()

        df['paid_user'] = 0
        df.loc[df['user_id'].isin(paid_user),'paid_user'] = 1

        self.train = df[:len(self.train)]
        self.valid = df[len(self.train):]
        self.train = self.train.reset_index(drop=True)
        self.valid = self.valid.reset_index(drop=True)

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]


class USER_ID_LECTURE(Feature):
    def user_lecture_part_past_features(self,df,lecture_part_num_dict,part):
        count = np.zeros(len(df), dtype=np.int32)
        for cnt,row in enumerate(tqdm(df[['user_id','content_type_id','part']].values)):
            count[cnt] = lecture_part_num_dict[row[0]]
            if row[2] == part:
                lecture_part_num_dict[row[0]] += row[1]
        user_feats_df = pd.DataFrame({f'lecture_part{part}_count':count})
        df = pd.concat([df, user_feats_df], axis=1)
        return df

    def user_lecture_past_features(self,df,lecture_part_num_dict):
        count = np.zeros(len(df), dtype=np.int32)
        for cnt,row in enumerate(tqdm(df[['user_id','content_type_id']].values)):
            count[cnt] = lecture_part_num_dict[row[0]]
            lecture_part_num_dict[row[0]] += row[1]
        user_feats_df = pd.DataFrame({f'lecture_count':count})
        df = pd.concat([df, user_feats_df], axis=1)
        return df

    def create_features(self):
        self.train = pd.read_feather('./data/input/train.feather')
        train_index = pd.read_feather(f'./data/train_valid/{CV}_train.feather')
        valid_index = pd.read_feather(f'./data/train_valid/{CV}_valid.feather')
        create_feats = ['lecture_count','lecture_part1_count','lecture_part2_count','lecture_part3_count','lecture_part4_count','lecture_part5_count','lecture_part6_count','lecture_part7_count']

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
        self.train = self.train.sort_values('row_id').reset_index(drop=True)

        del qs_train,lc_train
        gc.collect()


        lecture_part_num_dict_list = [defaultdict(int) for _ in range(8)]
        for part in range(7):
            print(f'part:{part}')
            self.train = self.user_lecture_part_past_features(self.train,lecture_part_num_dict_list[part],part+1)

        self.train = self.user_lecture_past_features(self.train,lecture_part_num_dict_list[7])

        self.valid = self.train[self.train['row_id'].isin(valid_index['row_id'])]
        self.train = self.train[self.train['row_id'].isin(train_index['row_id'])]


        self.train = self.train[self.train['content_type_id'] == 0].reset_index(drop=True)
        self.valid = self.valid[self.valid['content_type_id'] == 0].reset_index(drop=True)
        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]


        # self.test = self.test[self.test['content_type_id'] == 0].reset_index(drop=True)
        user_lecture_part_df = pd.DataFrame()
        user_lecture_part_df['user_id'] = list(lecture_part_num_dict_list[0].keys())
        for part in range(7):
            user_lecture_part_df[f'lecture_part{part+1}_count'] = list(lecture_part_num_dict_list[part].values())
        user_lecture_part_df['lecture_count'] = list(lecture_part_num_dict_list[7].values())
        user_lecture_part_df.to_feather(f'./{Feature.kernel_dir}/lecture_part_num.feather')

        if not MINI_DATA:
            for part in range(8):
                with open(f'./{Feature.kernel_dir}/lecture_part{part}_num_dict.pkl','wb') as f:
                    pickle.dump(lecture_part_num_dict_list[part],f)


class LOOP(Feature):

    def update_time_stamp_user(self,timestamp_u,timestamp_u_recency_1,timestamp_u_recency_2,timestamp_u_recency_3,timestamp_u_incorrect,timestamp_u_incorrect_recency):
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

        if len(timestamp_u_incorrect[row[0]]) == 0:
            timestamp_u_incorrect_recency[num] = np.nan
        else:
            timestamp_u_incorrect_recency[num] = row[5] - timestamp_u_incorrect[row[0]][0]


    def add_past_feature(self,df, answered_correctly_u_count, answered_correctly_u_sum, elapsed_time_u_sum, explanation_u_sum, timestamp_u,
                        timestamp_u_incorrect, answered_correctly_q_count, answered_correctly_q_sum,
                        elapsed_time_q_sum, explanation_q_sum, answered_correctly_uq, update = True):
        answered_correctly_u_avg = np.zeros(len(df), dtype = np.float32)
        elapsed_time_u_avg = np.zeros(len(df), dtype = np.float32)
        explanation_u_avg = np.zeros(len(df), dtype = np.float32)
        timestamp_u_recency_1 = np.zeros(len(df), dtype = np.float32)
        timestamp_u_recency_2 = np.zeros(len(df), dtype = np.float32)
        timestamp_u_recency_3 = np.zeros(len(df), dtype = np.float32)
        timestamp_u_incorrect_recency = np.zeros(len(df), dtype = np.float32)
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



        for num, row in enumerate(tqdm(df[['user_id', 'answered_correctly', 'content_id',
                                        'prior_question_elapsed_time', 'prior_question_had_explanation', 'timestamp','tags1']].values)):

            # Client features assignation
            # ------------------------------------------------------------------
            if answered_correctly_u_count[row[0]] != 0:
                answered_correctly_u_avg[num] = answered_correctly_u_sum[row[0]] / answered_correctly_u_count[row[0]]
                elapsed_time_u_avg[num] = elapsed_time_u_sum[row[0]] / answered_correctly_u_count[row[0]]
                explanation_u_avg[num] = explanation_u_sum[row[0]] / answered_correctly_u_count[row[0]]
            else:
                answered_correctly_u_avg[num] = np.nan
                elapsed_time_u_avg[num] = np.nan
                explanation_u_avg[num] = np.nan

            # update time_stamp_user
            self.update_time_stamp_user(timestamp_u,timestamp_u_recency_1,timestamp_u_recency_2,timestamp_u_recency_3,timestamp_u_incorrect,timestamp_u_incorrect_recency)

            # ------------------------------------------------------------------
            # Question features assignation
            if answered_correctly_q_count[row[2]] != 0:
                answered_correctly_q_avg[num] = answered_correctly_q_sum[row[2]] / answered_correctly_q_count[row[2]]
                elapsed_time_q_avg[num] = elapsed_time_q_sum[row[2]] / answered_correctly_q_count[row[2]]
                explanation_q_avg[num] = explanation_q_sum[row[2]] / answered_correctly_q_count[row[2]]
            else:
                answered_correctly_q_avg[num] = np.nan
                elapsed_time_q_avg[num] = np.nan
                explanation_q_avg[num] = np.nan

            # ------------------------------------------------------------------
            # tags1 features assignation
            if answered_correctly_tags1_count[row[6]] != 0:
                answered_correctly_tags1_avg[num] = answered_correctly_tags1_sum[row[2]] / answered_correctly_tags1_count[row[2]]
            else:
                answered_correctly_tags1_avg[num] == np.nan


            # ------------------------------------------------------------------
            # Client Question assignation
            answered_correctly_uq_count[num] = answered_correctly_uq[row[0]][row[2]]
            # ------------------------------------------------------------------
            # ------------------------------------------------------------------
            # Client features updates
            answered_correctly_u_count[row[0]] += 1
            elapsed_time_u_sum[row[0]] += row[3]
            explanation_u_sum[row[0]] += int(row[4])
            if len(timestamp_u[row[0]]) == 3:
                timestamp_u[row[0]].pop(0)
                timestamp_u[row[0]].append(row[5])
            else:
                timestamp_u[row[0]].append(row[5])
            # ------------------------------------------------------------------
            # Question features updates
            answered_correctly_q_count[row[2]] += 1
            elapsed_time_q_sum[row[2]] += row[3]
            explanation_q_sum[row[2]] += int(row[4])
            # ------------------------------------------------------------------
            # Client Question updates
            answered_correctly_uq[row[0]][row[2]] += 1

            # ------------------------------------------------------------------
            # tags1 updates


            # ------------------------------------------------------------------
            # Flag for training and inference
            if True:
                # ------------------------------------------------------------------
                # Client features updates
                answered_correctly_u_sum[row[0]] += row[1]
                if row[1] == 0:
                    if len(timestamp_u_incorrect[row[0]]) == 1:
                        timestamp_u_incorrect[row[0]].pop(0)
                        timestamp_u_incorrect[row[0]].append(row[5])
                    else:
                        timestamp_u_incorrect[row[0]].append(row[5])

                # ------------------------------------------------------------------
                # Question features updates
                answered_correctly_q_sum[row[2]] += row[1]
                # ------------------------------------------------------------------


        user_df = pd.DataFrame({'answered_correctly_user_avg': answered_correctly_u_avg, 'elapsed_time_user_avg': elapsed_time_u_avg, 'explanation_user_avg': explanation_u_avg,
                                'answered_correctly_q_avg': answered_correctly_q_avg, 'elapsed_time_q_avg': elapsed_time_q_avg, 'explanation_q_avg': explanation_q_avg,
                                'answered_correctly_uq_count': answered_correctly_uq_count, 'timestamp_u_recency_1': timestamp_u_recency_1, 'timestamp_u_recency_2': timestamp_u_recency_2,
                                'timestamp_u_recency_3': timestamp_u_recency_3, 'timestamp_u_incorrect_recency': timestamp_u_incorrect_recency})

        df = pd.concat([df, user_df], axis = 1)
        return df

    def create_features(self):
        create_feats = ['answered_correctly_user_avg','elapsed_time_user_avg','explanation_user_avg',\
                        'answered_correctly_q_avg','elapsed_time_q_avg','explanation_q_avg',\
                        'answered_correctly_uq_count','timestamp_u_recency_1','timestamp_u_recency_2',\
                        'timestamp_u_recency_3','timestamp_u_incorrect_recency']

        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')

        answered_correctly_u_count = defaultdict(int)
        answered_correctly_u_sum = defaultdict(int)
        elapsed_time_u_sum = defaultdict(int)
        explanation_u_sum = defaultdict(int)
        timestamp_u = defaultdict(list)
        timestamp_u_incorrect = defaultdict(list)

        # Question dictionaries
        answered_correctly_q_count = defaultdict(int)
        answered_correctly_q_sum = defaultdict(int)
        elapsed_time_q_sum = defaultdict(int)
        explanation_q_sum = defaultdict(int)

        # tags dict
        answered_correctly_tags1_count = defaultdict(int)
        answered_correctly_tags1_sum = defaultdict(int)

        # Client Question dictionary
        answered_correctly_uq = defaultdict(lambda: defaultdict(int))

        self.train = self.add_past_feature(self.train, answered_correctly_u_count, answered_correctly_u_sum, elapsed_time_u_sum, explanation_u_sum,
                                            timestamp_u, timestamp_u_incorrect, answered_correctly_q_count, answered_correctly_q_sum,
                                            elapsed_time_q_sum, explanation_q_sum, answered_correctly_uq)

        self.valid = self.add_past_feature(self.valid, answered_correctly_u_count, answered_correctly_u_sum, elapsed_time_u_sum, explanation_u_sum,
                                            timestamp_u, timestamp_u_incorrect, answered_correctly_q_count, answered_correctly_q_sum,
                                            elapsed_time_q_sum, explanation_q_sum, answered_correctly_uq)

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]

        if not MINI_DATA:
            features_dicts = {
                            'answered_correctly_user_count': answered_correctly_u_count,
                            'answered_correctly_user_sum': answered_correctly_u_sum,
                            'elapsed_time_user_sum': elapsed_time_u_sum,
                            'explanation_user_sum': explanation_u_sum,
                            'answered_correctly_q_count': answered_correctly_q_count,
                            'answered_correctly_q_sum': answered_correctly_q_sum,
                            'elapsed_time_q_sum': elapsed_time_q_sum,
                            'explanation_q_sum': explanation_q_sum,
                            'answered_correctly_uq': answered_correctly_uq,
                            'timestamp_u': timestamp_u,
                            'timestamp_u_incorrect': timestamp_u_incorrect
            }
            with open('./features/kernel_data/loop_feats.pkl','wb') as f:
                pickle.dump(features_dicts,features_dicts)


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