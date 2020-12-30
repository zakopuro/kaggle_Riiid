import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import os
import time
import sys
import gc
sys.path.append('./')
from utils import data_util,logger


path = 'all_data'

SEED = 123
FEATURES_LIST = ['BASE_FIX','LOOP_FIX_TIME2','TAGS','GROUP_BY','BUNDLE_ID','ROLLING_MEAN2','ROLLING_PART_MEAN3']

# BASE = ['part','prior_question_had_explanation','prior_question_elapsed_time']
# LOOP = ['ans_user_avg','elapsed_time_user_avg','explanation_user_avg',
#         'ans_content_avg','elapsed_time_content_avg','explanation_content_avg',
#         'lag_time_1','lag_time_2','lag_time_3','lag_incorrect_time',
#         'ans_user_content_count']

USE_COLS = ['prior_question_elapsed_time', 'prior_question_had_explanation', 'part',
            'ans_user_avg', 'ans_user_count','elapsed_time_user_avg', 'explanation_user_avg','elapsed_time_user_sum',
            'ans_content_avg', 'elapsed_time_content_avg', 'explanation_content_avg',
            'user_content_count',
            'lag_time_1', 'lag_time_2', 'lag_time_3','lag_incorrect_time',
            'lag_part_time_1','lag_part_time_2','lag_part_time_3',
            "tags_pca_0", "tags_pca_1",'tags1','tags2','tags3','tags_nan_count','tags_nan_count_mean','tags1_cut_mean','tags1_cut',
            'first_bundle',
            'ans_part_mean','ans_user_part_avg','user_part_count',
            'ans_community_mean','community',
            'content_id','bundle_id',
            'first_bundle_cut','first_bundle_cut_mean',
            'rolling_mean_10','rolling_mean_3','rolling_part_mean_10','rolling_part_mean_3']


# GROUP_BY =['paid_user_part_mean']


# USE_COLS = BASE + LOOP

TARGET = 'answered_correctly'

CAT_FEATURES = ['part','community']


def run_lgb(train,valid,LOG):
    # lgb_params = {
    #             'n_estimators': 24000,
    #             'objective': 'binary',
    #             'boosting_type': 'gbdt',
    #             'metric': 'auc',
    #             'max_depth': 7,
    #             'learning_rate': 0.2,
    #             'seed': 127,
    #             'early_stopping_rounds': 50
    #         }
    lgb_params = {'objective': 'binary',
              'seed': SEED,
              'metric': 'auc',
              'num_leaves': 200,
              'feature_fraction': 0.75,
              'bagging_freq': 10,
              'bagging_fraction': 0.80
             }


    train_x,valid_x = train[USE_COLS],valid[USE_COLS]
    train_y,valid_y = train[TARGET],valid[TARGET]
    del train,valid
    gc.collect()

    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_eval = lgb.Dataset(valid_x, valid_y)
    LOG.info(f'start lgb train')
    t0 = time.time()
    model = lgb.train(lgb_params, lgb_train, valid_sets=[lgb_train, lgb_eval], verbose_eval=10,
                    early_stopping_rounds = 30,
                    num_boost_round = 10000)
    LOG.info(f'end lgb train : {time.time() - t0} s')

    val_pred = model.predict(valid_x)
    score = roc_auc_score(valid_y, val_pred)
    LOG.info(f"AUC = {score}")

    fi = pd.DataFrame()
    fi['features'] = train_x.columns.values.tolist()
    fi['importance'] = model.feature_importance(importance_type="gain")

    return model,fi,val_pred


def main():

    file_name = os.path.basename(__file__)[:-3]
    LOG = logger.Logger(name=f'{file_name}',filename=file_name)
    LOG.info('base line')
    LOG.info(f'{USE_COLS}')

    train = data_util.load_features(FEATURES_LIST,path=f'features/{path}',train_valid='train')
    valid = data_util.load_features(FEATURES_LIST,path=f'features/{path}',train_valid='valid')

    train = train.sample(15000000, random_state = SEED)

    # qs = pd.read_csv('./data/output/question_lsi50.csv')
    # qs = qs[['question_id','tags_lsi']]
    # qs = qs.rename(columns={'question_id':'content_id'})
    # train = pd.merge(train,qs,on='content_id',how='left')
    # valid = pd.merge(valid,qs,on='content_id',how='left')

    # train = data_util.reduce_mem_usage(train)
    # valid = data_util.reduce_mem_usage(valid)

    LOG.info(f'train_size:{train[USE_COLS].shape} valid_size:{valid[USE_COLS].shape}')

    model,fi,valid['pred'] = run_lgb(train=train,valid=valid,LOG=LOG)
    data_util.seve_model(model,fi,file_name)

    valid[['row_id','pred']].to_feather(f'./data/oof/{file_name}.feather')





if __name__ == "__main__":
    main()