import pandas as pd
import numpy as np
import gc
import os
from tqdm import tqdm
from collections import defaultdict
from base_create_features import Feature, get_arguments, generate_features
from pathlib import Path
import pickle

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
        qs_lc = pd.concat([qs,lc])

        self.train.loc[self.train['prior_question_had_explanation'] == False , 'prior_question_had_explanation'] = 0
        self.train.loc[self.train['prior_question_had_explanation'] == True , 'prior_question_had_explanation'] = 1
        self.train['prior_question_had_explanation'] = self.train['prior_question_had_explanation'].astype(float)

        self.train = pd.merge(self.train,qs_lc,on='content_id',how='left')
        for i in range(1,7):
            self.train[f'tags{i}'] = self.train[f'tags{i}'].astype(float)

        self.test = self.train.copy()
        self.valid = self.train[self.train['row_id'].isin(valid_index['row_id'])]
        self.train = self.train[self.train['row_id'].isin(train_index['row_id'])]


        self.train = self.train[self.train['content_type_id'] == 0].reset_index(drop=True)
        self.valid = self.valid[self.valid['content_type_id'] == 0].reset_index(drop=True)
        self.test = self.test[self.test['content_type_id'] == 0].reset_index(drop=True)



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
        create_feats = ['answered_correctly_avg_tags1','answered_correctly_sum_tags1','answered_correctly_std_tags1']
        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')



if __name__ == "__main__":
    args = get_arguments()
    generate_features(globals(),args.force)