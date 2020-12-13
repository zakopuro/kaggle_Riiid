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

MINI_DATA = True
if MINI_DATA:
    path = 'mini_data'
else:
    path = 'data'


FEATURES_LIST = ['BASE','USER_ID','CONTENT','BUNDLE','TASK_CONTAINER','PART','TAGS1','TIME','USER_ID_LECTURE']

USE_COLS = ['prior_question_elapsed_time','prior_question_had_explanation','bundle_id','part','tags1','tags2','tags3','l_type_of','content_id','task_container_id','small_quenstion',\
            'answered_correctly_avg_user','answered_correctly_sum_user','count_user','answered_correctly_avg_content','answered_correctly_sum_content','answered_correctly_std_content','content_num',\
            'answered_correctly_avg_bundle','answered_correctly_sum_bundle','answered_correctly_std_bundle','bundle_num',\
            'answered_correctly_avg_task_container','answered_correctly_sum_task_container','answered_correctly_std_task_container',\
            'answered_correctly_avg_part','answered_correctly_sum_part','answered_correctly_std_part',\
            'answered_correctly_avg_tags1','answered_correctly_sum_tags1','answered_correctly_std_tags1',\
            'lag_time','user_lag_time_mean','days_elapsed','lag_days','paid_user','lag_hours',\
            'lecture_count','lecture_part1_count','lecture_part2_count','lecture_part3_count','lecture_part4_count','lecture_part5_count','lecture_part6_count','lecture_part7_count']

TARGET = 'answered_correctly'

CAT_FEATURES = ['part','tags1','tags2','tags3','content_id','task_container_id','prior_question_had_explanation',\
                'bundle_id']


def run_lgb(train,valid,LOG):
    lgb_params = {
                'n_estimators': 24000,
                'objective': 'binary',
                'boosting_type': 'gbdt',
                'metric': 'auc',
                'max_depth': 7,
                'learning_rate': 0.08,
                'seed': 127,
                'early_stopping_rounds': 50
            }
    train_x,valid_x = train[USE_COLS],valid[USE_COLS]
    train_y,valid_y = train[TARGET],valid[TARGET]
    del train,valid
    gc.collect()

    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_eval = lgb.Dataset(valid_x, valid_y)
    LOG.info(f'start lgb train')
    t0 = time.time()
    model = lgb.train(lgb_params, lgb_train, valid_sets=[lgb_train, lgb_eval], verbose_eval=10,categorical_feature=CAT_FEATURES)
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

    # train = data_util.reduce_mem_usage(train)
    # valid = data_util.reduce_mem_usage(valid)

    LOG.info(f'train_size:{train.shape} valid_size:{valid.shape}')

    model,fi,valid_pred = run_lgb(train=train,valid=valid,LOG=LOG)
    data_util.seve_model(model,fi,file_name)





if __name__ == "__main__":
    main()