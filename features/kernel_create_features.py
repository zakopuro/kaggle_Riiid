import pandas as pd
import numpy as np
import pickle
import gc
import os
from tqdm import tqdm
from collections import defaultdict
from base_create_features import Feature, get_arguments, generate_features
from pathlib import Path
Feature.dir = 'features/kernel_data'

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

        self.train.to_feather(f'./{Feature.dir}/BASE_train.feather')

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
        answered_correctly_sum_u_dict = defaultdict(int)
        count_u_dict = defaultdict(int)
        self.train = self.user_past_features(self.train, answered_correctly_sum_u_dict, count_u_dict)

        with open(f'./{Feature.dir}/user_dict_sum.pkl','wb') as f:
            pickle.dump(answered_correctly_sum_u_dict,f)
        with open(f'./{Feature.dir}/user_dict_count.pkl','wb') as f:
            pickle.dump(count_u_dict,f)





class PART(Feature):

    def create_features(self):
        # 少しリークしているが、これくらいならOK判断
        create_feats = ['answered_correctly_avg_part','reading_part','answered_correctly_avg_reading_part']
        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.train['reading_part'] = 0
        self.train.loc[self.train['part'] >=5,'reading_part'] = 1

        part_mean = self.train[['part',TARGET]].groupby(['part']).agg(['mean'])
        part_mean.columns = ['answered_correctly_avg_part']
        read_mean = self.train[['reading_part',TARGET]].groupby(['reading_part']).agg(['mean'])
        read_mean.columns = ['answered_correctly_avg_reading_part']

        part_mean = part_mean.reset_index()
        read_mean = read_mean.reset_index()

        part_mean.to_feather(f'./{Feature.dir}/part_mean.feather')
        read_mean.to_feather(f'./{Feature.dir}/read)part_mean.feather')


        # self.train = self.train[create_feats]


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
        self.train['user_id_part'] = self.train['user_id'].astype(str) + '-' + self.train['part'].astype(str)
        answered_correctly_sum_u_p_dict = defaultdict(int)
        count_u_p_dict = defaultdict(int)

        self.train = self.user_part_past_features(self.train, answered_correctly_sum_u_p_dict, count_u_p_dict)

        with open(f'./{Feature.dir}/user_part_dict_sum.pkl','wb') as f:
            pickle.dump(answered_correctly_sum_u_p_dict,f)
        with open(f'./{Feature.dir}/user_part_dict_count.pkl','wb') as f:
            pickle.dump(count_u_p_dict,f)



class CONTENT(Feature):

    def create_features(self):
        create_feats = ['answered_correctly_avg_content','answered_correctly_sum_content','content_num']
        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')

        content_ans_av_sum = self.train[['content_id',TARGET]].groupby('content_id')[TARGET].agg(['mean','sum'])
        content_ans_av_sum.columns = ['answered_correctly_avg_content','answered_correctly_sum_content']

        content_num = pd.DataFrame(self.train['content_id'].value_counts().sort_values(ascending=False)).reset_index()
        content_num.columns = ['content_id','content_num']


        content_ans_av_sum = content_ans_av_sum.reset_index()
        content_ans_av_sum.to_feather(f'./{Feature.dir}/content_mean.feather')

        content_num = content_num.reset_index()
        content_num.to_feather(f'./{Feature.dir}/content_num.feather')

        # self.train = pd.merge(self.train,content_num,on='content_id',how='left')



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
        self.train['user_id_content'] = self.train['user_id'].astype(str) + '-' + self.train['content_id'].astype(str)

        answered_correctly_sum_u_c_dict = defaultdict(int)
        count_u_c_dict = defaultdict(int)

        self.train = self.user_content_past_features(self.train, answered_correctly_sum_u_c_dict, count_u_c_dict)

        with open(f'./{Feature.dir}/user_content_dict_sum.pkl','wb') as f:
            pickle.dump(answered_correctly_sum_u_c_dict,f)
        with open(f'./{Feature.dir}/user_content_dict_count.pkl','wb') as f:
            pickle.dump(count_u_c_dict,f)


class USER_BUNDLE(Feature):
    def user_content_past_features(self,df,answered_correctly_sum_u_c_dict, count_u_c_dict):
        acsu = np.zeros(len(df), dtype=np.int32)
        cu = np.zeros(len(df), dtype=np.int32)
        for cnt,row in enumerate(tqdm(df[['user_id_bundle',TARGET]].values)):
            acsu[cnt] = answered_correctly_sum_u_c_dict[row[0]]
            cu[cnt] = count_u_c_dict[row[0]]
            answered_correctly_sum_u_c_dict[row[0]] += row[1]
            count_u_c_dict[row[0]] += 1
        user_feats_df = pd.DataFrame({'answered_correctly_sum_user_bundle':acsu, 'count_user_bundle':cu})
        user_feats_df['answered_correctly_avg_user_bundle'] = user_feats_df['answered_correctly_sum_user_bundle'] / user_feats_df['count_user_bundle']
        df = pd.concat([df, user_feats_df], axis=1)
        return df


    def create_features(self):
        create_feats = ['answered_correctly_avg_user_bundle','answered_correctly_sum_user_bundle','count_user_bundle']

        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.train['user_id_bundle'] = self.train['user_id'].astype(str) + '-' + self.train['bundle_id'].astype(str)
        answered_correctly_sum_u_c_dict = defaultdict(int)
        count_u_c_dict = defaultdict(int)

        self.train = self.user_content_past_features(self.train, answered_correctly_sum_u_c_dict, count_u_c_dict)

        with open(f'./{Feature.dir}/user_bundle_dict_sum.pkl','wb') as f:
            pickle.dump(answered_correctly_sum_u_c_dict,f)
        with open(f'./{Feature.dir}/user_bundle_dict_count.pkl','wb') as f:
            pickle.dump(count_u_c_dict,f)

class USER_TAGS(Feature):
    def user_content_past_features(self,df,answered_correctly_sum_u_c_dict, count_u_c_dict):
        acsu = np.zeros(len(df), dtype=np.int32)
        cu = np.zeros(len(df), dtype=np.int32)
        for cnt,row in enumerate(tqdm(df[['user_id_tags1',TARGET]].values)):
            acsu[cnt] = answered_correctly_sum_u_c_dict[row[0]]
            cu[cnt] = count_u_c_dict[row[0]]
            answered_correctly_sum_u_c_dict[row[0]] += row[1]
            count_u_c_dict[row[0]] += 1
        user_feats_df = pd.DataFrame({'answered_correctly_sum_user_tags1':acsu, 'count_user_tags1':cu})
        user_feats_df['answered_correctly_avg_user_tags1'] = user_feats_df['answered_correctly_sum_user_tags1'] / user_feats_df['count_user_tags1']
        df = pd.concat([df, user_feats_df], axis=1)
        return df


    def create_features(self):
        create_feats = ['answered_correctly_avg_user_tags1','answered_correctly_sum_user_tags1','count_user_tags1']

        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.train['user_id_tags1'] = self.train['user_id'].astype(str) + '-' + self.train['tags1'].astype(str)

        answered_correctly_sum_u_c_dict = defaultdict(int)
        count_u_c_dict = defaultdict(int)

        self.train = self.user_content_past_features(self.train, answered_correctly_sum_u_c_dict, count_u_c_dict)
        with open(f'./{Feature.dir}/user_tags_dict_sum.pkl','wb') as f:
            pickle.dump(answered_correctly_sum_u_c_dict,f)
        with open(f'./{Feature.dir}/user_tasg_dict_count.pkl','wb') as f:
            pickle.dump(count_u_c_dict,f)


class TAGS(Feature):

    def create_features(self):
        create_feats = ['answered_correctly_avg_tags1']
        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')

        tag1_mean = self.train[['tags1',TARGET]].groupby(['tags1']).agg(['mean'])
        tag1_mean.columns = ['answered_correctly_avg_tags1']

        tag1_mean = tag1_mean.reset_index()
        tag1_mean.to_feather(f'./{Feature.dir}/tag1_mean.feather')


class USER_READING_PART(Feature):
    def user_reading_part_past_features(self,df,answered_correctly_sum_u_c_dict, count_u_c_dict):
        acsu = np.zeros(len(df), dtype=np.int32)
        cu = np.zeros(len(df), dtype=np.int32)
        for cnt,row in enumerate(tqdm(df[['user_id_reading_part',TARGET]].values)):
            acsu[cnt] = answered_correctly_sum_u_c_dict[row[0]]
            cu[cnt] = count_u_c_dict[row[0]]
            answered_correctly_sum_u_c_dict[row[0]] += row[1]
            count_u_c_dict[row[0]] += 1
        user_feats_df = pd.DataFrame({'answered_correctly_sum_user_reading_part':acsu, 'count_user_reading_part':cu})
        user_feats_df['answered_correctly_avg_user_reading_part'] = user_feats_df['answered_correctly_sum_user_reading_part'] / user_feats_df['count_user_reading_part']
        df = pd.concat([df, user_feats_df], axis=1)
        return df


    def create_features(self):
        create_feats = ['answered_correctly_avg_user_reading_part','answered_correctly_sum_user_reading_part','count_user_reading_part']

        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.train['reading_part'] = 0
        self.train.loc[self.train['part'] >=5,'reading_part'] = 1

        self.train['user_id_reading_part'] = self.train['user_id'].astype(str) + '-' + self.train['reading_part'].astype(str)
        answered_correctly_sum_u_c_dict = defaultdict(int)
        count_u_c_dict = defaultdict(int)

        self.train = self.user_reading_part_past_features(self.train, answered_correctly_sum_u_c_dict, count_u_c_dict)

        with open(f'./{Feature.dir}/user_reading_part_dict_sum.pkl','wb') as f:
            pickle.dump(answered_correctly_sum_u_c_dict,f)
        with open(f'./{Feature.dir}/user_reading_part_dict_count.pkl','wb') as f:
            pickle.dump(count_u_c_dict,f)


class TYPE_OF(Feature):
    def create_features(self):
        create_feats = ['answered_correctly_avg_type_of']
        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')

        self.train['type_of'] = self.train['type_of'].fillna('NAN')

        type_mean = self.train[['type_of',TARGET]].groupby('type_of').agg(['mean'])
        type_mean.columns = ['answered_correctly_avg_type_of']

        type_mean = type_mean.reset_index()
        type_mean.to_feather(f'./{Feature.dir}/type_mean.feather')


class USER_TYPE_OF(Feature):
    def user_type_of_part_past_features(self,df,answered_correctly_sum_u_c_dict, count_u_c_dict):
        acsu = np.zeros(len(df), dtype=np.int32)
        cu = np.zeros(len(df), dtype=np.int32)
        for cnt,row in enumerate(tqdm(df[['user_id_type_of',TARGET]].values)):
            acsu[cnt] = answered_correctly_sum_u_c_dict[row[0]]
            cu[cnt] = count_u_c_dict[row[0]]
            answered_correctly_sum_u_c_dict[row[0]] += row[1]
            count_u_c_dict[row[0]] += 1
        user_feats_df = pd.DataFrame({'answered_correctly_sum_user_type_of':acsu, 'count_user_type_of':cu})
        user_feats_df['answered_correctly_avg_user_type_of'] = user_feats_df['answered_correctly_sum_user_type_of'] / user_feats_df['count_user_type_of']
        df = pd.concat([df, user_feats_df], axis=1)
        return df


    def create_features(self):
        create_feats = ['answered_correctly_avg_user_type_of','answered_correctly_sum_user_type_of','count_user_type_of']

        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')

        self.train['type_of'] = self.train['type_of'].fillna('NAN')

        self.train['user_id_type_of'] = self.train['user_id'].astype(str) + '-' + self.train['type_of'].astype(str)
        answered_correctly_sum_u_c_dict = defaultdict(int)
        count_u_c_dict = defaultdict(int)

        self.train = self.user_type_of_part_past_features(self.train, answered_correctly_sum_u_c_dict, count_u_c_dict)

        with open(f'./{Feature.dir}/user_type_of_dict_sum.pkl','wb') as f:
            pickle.dump(answered_correctly_sum_u_c_dict,f)
        with open(f'./{Feature.dir}/user_type_of_part_dict_count.pkl','wb') as f:
            pickle.dump(count_u_c_dict,f)



if __name__ == "__main__":
    args = get_arguments()
    generate_features(globals(),args.force)