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


FEATURES_LIST = ['BASE','LOOP']

BASE = ['part','prior_question_had_explanation','prior_question_elapsed_time']
LOOP = ['answered_correctly_user_avg','elapsed_time_user_avg','explanation_user_avg',\
        'answered_correctly_q_avg','elapsed_time_q_avg','explanation_q_avg',\
        'answered_correctly_uq_count','timestamp_u_recency_1','timestamp_u_recency_2',\
        'timestamp_u_recency_3','timestamp_u_incorrect_recency',\
        'answered_correctly_tags1_avg','elapsed_time_tags1_avg','explanation_tags1_avg']


USE_COLS = BASE + LOOP

TARGET = 'answered_correctly'

CAT_FEATURES = ['part']


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