import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import os
import time
import sys
sys.path.append('./')
from utils import data_util,logger


feature_list = ['BASE','PART','USER_ID','CONTENT']
USE_COLS_BASE = ['timestamp','content_id','content_type_id','task_container_id','prior_question_elapsed_time','prior_question_had_explanation','bundle_id',\
                'part','tags1','tags2','tags3','tags4','tags5','tags6']
USE_COLS_PART = ['user_part_mean','user_part_sum','part_ans_mean']
USE_COLS_USER_ID = ['past_correctly_sum','past_not_correctly_sum','past_correctly_mean']
USE_COLS_CONTENT = ['content_id_ans_mean','content_id_num']
USE_COLS = USE_COLS_BASE + USE_COLS_PART + USE_COLS_USER_ID + USE_COLS_CONTENT
TARGET = 'answered_correctly'

CAT_FEATURES = ['part','tags1','tags2','tags3','tags4','tags5','tags6','content_id','content_type_id','task_container_id','prior_question_had_explanation','bundle_id']


def run_xgb(train_x,train_y,valid_x,valid_y,LOG):
    xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric':'auc',
                'num_boost_round':24000,
                'max_depth':7,
                'learning_rate':0.08,
                'subsample':0.85,
                'colsample_bytree':0.85,
                'tree_method':'gpu_hist',  # THE MAGICAL PARAMETER
                # 'gpu_id':1,
                'reg_alpha':0.15,
                'reg_lamdba':0.85,
                'early_stopping_rounds':50
            }

    # lgb_train = lgb.Dataset(train_x, train_y)
    # lgb_eval = lgb.Dataset(valid_x, valid_y)
    xgb_train = xgb.DMatrix(train_x, label=train_y)
    xgb_eval = xgb.DMatrix(valid_x, label=valid_y)
    LOG.info('start lgb train')
    t0 = time.time()
    model = xgb.train(params=xgb_params, dtrain=xgb_train, evals=[(xgb_train,'Train'), (xgb_eval,'Val')], verbose_eval=10)
    LOG.info(f'end lgb train : {time.time() - t0} s')

    val_pred = model.predict(xgb_eval)
    score = roc_auc_score(valid_y, val_pred)
    LOG.info(f"AUC = {score}")

    # feature importance
    fi = pd.DataFrame()
    # fi['features'] = train_x.columns.values.tolist()
    # fi['importance'] = model.feature_importances_(importance_type="gain")

    return model,fi


def main():

    file_name = os.path.basename(__file__)[:-3]
    LOG = logger.Logger(name=f'{file_name}',filename=file_name)
    LOG.info('base line')
    LOG.info(f'{USE_COLS}')

    train_df = data_util.load_features(feature_list)
    train_df = train_df.sample(frac=0.10,random_state=127)
    valid_df = train_df.sample(frac=0.02,random_state=127)
    valid_id = valid_df['row_id']
    train_df = train_df[~train_df['row_id'].isin(valid_id)]

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    LOG.info(f'train shape : {train_df.shape}')
    LOG.info(f'valid shape : {valid_df.shape}')


    train_x = train_df[USE_COLS]
    train_y = train_df[TARGET]
    valid_x = valid_df[USE_COLS]
    valid_y = valid_df[TARGET]

    lgb_model,fi = run_xgb(train_x=train_x,train_y=train_y,valid_x=valid_x,valid_y=valid_y,LOG=LOG)

    data_util.seve_model(lgb_model,fi,file_name)





if __name__ == "__main__":
    main()