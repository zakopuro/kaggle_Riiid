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
        qs_lc = pd.concat([qs,lc])
        self.train = pd.merge(self.train,qs_lc,on='content_id',how='left')


        lecture_part_num_dict_list = [defaultdict(int) for _ in range(8)]
        for part in range(7):
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

class LECTURE_COUNT(Feature):

    def create_features(self):
        create_feats = ['lecture_part1_count_cut','answered_correctly_avg_lecture_count_part1','lecture_part2_count_cut','answered_correctly_avg_lecture_count_part2',\
                        'lecture_part3_count_cut','answered_correctly_avg_lecture_count_part3','lecture_part4_count_cut','answered_correctly_avg_lecture_count_part4',\
                        'lecture_part5_count_cut','answered_correctly_avg_lecture_count_part5','lecture_part6_count_cut','answered_correctly_avg_lecture_count_part6',\
                        'lecture_part7_count_cut','answered_correctly_avg_lecture_count_part7']

        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')

        lecture_count_train_df = pd.read_feather(f'./{Feature.dir}/USER_ID_LECTURE_train.feather')
        lecture_count_valid_df = pd.read_feather(f'./{Feature.dir}/USER_ID_LECTURE_valid.feather')

        self.train = pd.concat([self.train,lecture_count_train_df],axis=1)
        self.valid = pd.concat([self.valid,lecture_count_valid_df],axis=1)

        for part in range(1,8):
            self.train[f'lecture_part{part}_count_cut'] = 0
            self.valid[f'lecture_part{part}_count_cut'] = 0
            self.train.loc[self.train[f'lecture_part{part}_count'] >= 1, f'lecture_part{part}_count_cut'] = 1
            self.train.loc[self.train[f'lecture_part{part}_count'] >= 6, f'lecture_part{part}_count_cut'] = 2
            self.train.loc[self.train[f'lecture_part{part}_count'] >= 10, f'lecture_part{part}_count_cut'] = 3
            self.valid.loc[self.valid[f'lecture_part{part}_count'] >= 1, f'lecture_part{part}_count_cut'] = 1
            self.valid.loc[self.valid[f'lecture_part{part}_count'] >= 6, f'lecture_part{part}_count_cut'] = 2
            self.valid.loc[self.valid[f'lecture_part{part}_count'] >= 10, f'lecture_part{part}_count_cut'] = 3

        self.test = pd.concat([self.train,self.valid]).reset_index(drop=True)

        for part in range(1,8):
            lec_part_mean = self.train[[f'lecture_part{part}_count_cut',TARGET]].groupby(f'lecture_part{part}_count_cut').mean()
            lec_part_mean.columns = [f'answered_correctly_avg_lecture_count_part{part}']
            self.train = pd.merge(self.train,lec_part_mean,on=f'lecture_part{part}_count_cut',how='left')
            self.valid = pd.merge(self.valid,lec_part_mean,on=f'lecture_part{part}_count_cut',how='left')


        if not MINI_DATA:
            for part in range(1,8):
                lec_part_mean = self.test[[f'lecture_part{part}_count_cut',TARGET]].groupby(f'lecture_part{part}_count_cut').mean()
                lec_part_mean.columns = [f'answered_correctly_avg_lecture_count_part{part}']
                lec_part_mean = lec_part_mean.reset_index()
                lec_part_mean.to_feather(f'./{Feature.kernel_dir}/lecture_count_part{part}_mean.feather')


        self.test = pd.DataFrame()
        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]


class PART(Feature):

    def create_features(self):
        # 少しリークしているが、これくらいならOK判断
        create_feats = ['answered_correctly_avg_part','reading_part','answered_correctly_avg_reading_part']
        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')

        self.train['reading_part'] = 0
        self.valid['reading_part'] = 0
        self.train.loc[self.train['part'] >=5,'reading_part'] = 1
        self.valid.loc[self.valid['part'] >=5,'reading_part'] = 1

        part_mean = self.train[['part',TARGET]].groupby(['part']).agg(['mean'])
        part_mean.columns = ['answered_correctly_avg_part']
        read_mean = self.train[['reading_part',TARGET]].groupby(['reading_part']).agg(['mean'])
        read_mean.columns = ['answered_correctly_avg_reading_part']


        self.train = pd.merge(self.train, part_mean, on=['part'], how="left")
        self.valid = pd.merge(self.valid, part_mean, on=['part'], how="left")

        self.train = pd.merge(self.train, read_mean, on=['reading_part'], how="left")
        self.valid = pd.merge(self.valid, read_mean, on=['reading_part'], how="left")

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]

        if not MINI_DATA:
            self.test = pd.read_feather(f'./{Feature.kernel_dir}/BASE_test.feather')
            self.test['reading_part'] = 0
            self.test.loc[self.test['part'] >=5,'reading_part'] = 1
            part_mean_test = self.test[['part',TARGET]].groupby(['part']).agg(['mean'])
            part_mean_test.columns = ['answered_correctly_avg_part']
            read_mean_test = self.test[['reading_part',TARGET]].groupby(['reading_part']).agg(['mean'])
            read_mean_test.columns = ['answered_correctly_avg_reading_part']
            part_mean_test.to_feather(f'./{Feature.kernel_dir}/part_mean.feather')
            read_mean_test.to_feather(f'./{Feature.kernel_dir}/read_part_mean.feather')
            self.test = pd.DataFrame()

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

        if not MINI_DATA:
            with open(f'./{Feature.kernel_dir}/user_part_dict_sum.pkl','wb') as f:
                pickle.dump(answered_correctly_sum_u_p_dict,f)
            with open(f'./{Feature.kernel_dir}/user_part_dict_count.pkl','wb') as f:
                pickle.dump(count_u_p_dict,f)


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

        if not MINI_DATA:
            self.test = pd.read_feather(f'./{Feature.kernel_dir}/BASE_test.feather')

            content_ans_av_sum = self.test[['content_id',TARGET]].groupby('content_id')[TARGET].agg(['mean','sum'])
            content_ans_av_sum.columns = ['answered_correctly_avg_content','answered_correctly_sum_content']

            content_num = pd.DataFrame(self.test['content_id'].value_counts().sort_values(ascending=False)).reset_index()
            content_num.columns = ['content_id','content_num']

            content_ans_av_sum = content_ans_av_sum.reset_index()
            content_ans_av_sum.to_feather(f'./{Feature.kernel_dir}/content_mean.feather')

            content_num = content_num.reset_index()
            content_num.to_feather(f'./{Feature.kernel_dir}/content_num.feather')

            self.test = pd.DataFrame()


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

        if not MINI_DATA:
            with open(f'./{Feature.kernel_dir}/user_content_dict_sum.pkl','wb') as f:
                pickle.dump(answered_correctly_sum_u_c_dict,f)
            with open(f'./{Feature.kernel_dir}/user_content_dict_count.pkl','wb') as f:
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
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')
        self.train['user_id_bundle'] = self.train['user_id'].astype(str) + '-' + self.train['bundle_id'].astype(str)
        self.valid['user_id_bundle'] = self.valid['user_id'].astype(str) + '-' + self.valid['bundle_id'].astype(str)
        answered_correctly_sum_u_c_dict = defaultdict(int)
        count_u_c_dict = defaultdict(int)

        self.train = self.user_content_past_features(self.train, answered_correctly_sum_u_c_dict, count_u_c_dict)
        self.valid = self.user_content_past_features(self.valid, answered_correctly_sum_u_c_dict, count_u_c_dict)

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]

        if not MINI_DATA:
            with open(f'./{Feature.kernel_dir}/user_bundle_dict_sum.pkl','wb') as f:
                pickle.dump(answered_correctly_sum_u_c_dict,f)
            with open(f'./{Feature.kernel_dir}/user_bundle_dict_count.pkl','wb') as f:
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
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')
        self.train['user_id_tags1'] = self.train['user_id'].astype(str) + '-' + self.train['tags1'].astype(str)
        self.valid['user_id_tags1'] = self.valid['user_id'].astype(str) + '-' + self.valid['tags1'].astype(str)
        answered_correctly_sum_u_c_dict = defaultdict(int)
        count_u_c_dict = defaultdict(int)

        self.train = self.user_content_past_features(self.train, answered_correctly_sum_u_c_dict, count_u_c_dict)
        self.valid = self.user_content_past_features(self.valid, answered_correctly_sum_u_c_dict, count_u_c_dict)

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]

        if not MINI_DATA:
            with open(f'./{Feature.kernel_dir}/user_tags_dict_sum.pkl','wb') as f:
                pickle.dump(answered_correctly_sum_u_c_dict,f)
            with open(f'./{Feature.kernel_dir}/user_tags_dict_count.pkl','wb') as f:
                pickle.dump(count_u_c_dict,f)


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

        if not MINI_DATA:
            self.test = pd.read_feather(f'./{Feature.kernel_dir}/BASE_test.feather')

            tag1_mean = self.test[['tags1',TARGET]].groupby(['tags1']).agg(['mean'])
            tag1_mean.columns = ['answered_correctly_avg_tags1']

            tag1_mean = tag1_mean.reset_index()
            tag1_mean.to_feather(f'./{Feature.kernel_dir}/tag1_mean.feather')


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
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')
        reading_part_train_df = pd.read_feather(f'./{Feature.dir}/PART_train.feather')
        reading_part_valid_df = pd.read_feather(f'./{Feature.dir}/PART_valid.feather')

        self.train = pd.concat([self.train,reading_part_train_df],axis=1)
        self.valid = pd.concat([self.valid,reading_part_valid_df],axis=1)

        self.train['user_id_reading_part'] = self.train['user_id'].astype(str) + '-' + self.train['reading_part'].astype(str)
        self.valid['user_id_reading_part'] = self.valid['user_id'].astype(str) + '-' + self.valid['reading_part'].astype(str)
        answered_correctly_sum_u_c_dict = defaultdict(int)
        count_u_c_dict = defaultdict(int)

        self.train = self.user_reading_part_past_features(self.train, answered_correctly_sum_u_c_dict, count_u_c_dict)
        self.valid = self.user_reading_part_past_features(self.valid, answered_correctly_sum_u_c_dict, count_u_c_dict)

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]

        if not MINI_DATA:
            with open(f'./{Feature.kernel_dir}/user_reading_part_dict_sum.pkl','wb') as f:
                pickle.dump(answered_correctly_sum_u_c_dict,f)
            with open(f'./{Feature.kernel_dir}/user_reading_part_dict_count.pkl','wb') as f:
                pickle.dump(count_u_c_dict,f)


class TYPE_OF(Feature):
    def create_features(self):
        create_feats = ['answered_correctly_avg_type_of']
        self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')

        self.train['type_of'] = self.train['type_of'].fillna('NAN')
        self.valid['type_of'] = self.valid['type_of'].fillna('NAN')

        type_mean = self.train[['type_of',TARGET]].groupby('type_of').agg(['mean'])
        type_mean.columns = ['answered_correctly_avg_type_of']

        self.train = pd.merge(self.train,type_mean,on='type_of',how='left')
        self.valid = pd.merge(self.valid,type_mean,on='type_of',how='left')

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]

        if not MINI_DATA:
            self.test = pd.read_feather(f'./{Feature.dir}/BASE_test.feather')
            self.test['type_of'] = self.test['type_of'].fillna('NAN')

            type_mean = self.test[['type_of',TARGET]].groupby('type_of').agg(['mean'])
            type_mean.columns = ['answered_correctly_avg_type_of']

            type_mean = type_mean.reset_index()
            type_mean.to_feather(f'./{Feature.dir}/type_mean.feather')

            self.test = pd.DataFrame()

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
        self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')

        self.train['type_of'] = self.train['type_of'].fillna('NAN')
        self.valid['type_of'] = self.valid['type_of'].fillna('NAN')

        self.train['user_id_type_of'] = self.train['user_id'].astype(str) + '-' + self.train['type_of'].astype(str)
        self.valid['user_id_type_of'] = self.valid['user_id'].astype(str) + '-' + self.valid['type_of'].astype(str)
        answered_correctly_sum_u_c_dict = defaultdict(int)
        count_u_c_dict = defaultdict(int)

        self.train = self.user_type_of_part_past_features(self.train, answered_correctly_sum_u_c_dict, count_u_c_dict)
        self.valid = self.user_type_of_part_past_features(self.valid, answered_correctly_sum_u_c_dict, count_u_c_dict)

        self.train = self.train[create_feats]
        self.valid = self.valid[create_feats]

        if not MINI_DATA:
            with open(f'./{Feature.kernel_dir}/user_type_of_dict_sum.pkl','wb') as f:
                pickle.dump(answered_correctly_sum_u_c_dict,f)
            with open(f'./{Feature.kernel_dir}/user_type_of_part_dict_count.pkl','wb') as f:
                pickle.dump(count_u_c_dict,f)

# class CATEGORICAL(Feature):

#     def create_features(self):
#         create_feats = ['user_id_part','user_id_content','user_id_bundle','user_id_tags1','user_id_reading_part','user_id_type_of']

#         self.train = pd.read_feather(f'./{Feature.dir}/BASE_train.feather')
#         self.valid = pd.read_feather(f'./{Feature.dir}/BASE_valid.feather')

#         train_len = len(self.train)

#         df = pd.concat([self.train,self.valid])
#         df['reading_part'] = 0
#         df.loc[df['part'] >=5,'reading_part'] = 1

#         feats = ['type_of','content_id','bundle_id','tags1','reading_part','type_of']
#         for feat in feats:
#             df[feat] = df[feat].fillna('NAN')

#         df['user_id_part'] = df['user_id'].astype(str) + '-' + df['part'].astype(str)
#         df['user_id_content'] = df['user_id'].astype(str) + '-' + df['content_id'].astype(str)
#         df['user_id_bundle'] = df['user_id'].astype(str) + '-' + df['bundle_id'].astype(str)
#         df['user_id_tags1'] = df['user_id'].astype(str) + '-' + df['tags1'].astype(str)
#         df['user_id_reading_part'] = df['user_id'].astype(str) + '-' + df['reading_part'].astype(str)
#         df['user_id_type_of'] = df['user_id'].astype(str) + '-' + df['type_of'].astype(str)

#         for feats in create_feats:
#             df[feats] = _label_encoder(df[feats])

#         self.train = df[:train_len]
#         self.valid = df[train_len:]
#         del df
#         gc.collect()

#         self.train = self.train[create_feats]
#         self.valid = self.valid[create_feats]




if __name__ == "__main__":
    args = get_arguments()
    generate_features(globals(),args.force)