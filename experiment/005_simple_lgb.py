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


feature_list = ['BASE','PART','USER_ID','CONTENT']
USE_COLS_BASE = ['content_id','task_container_id','prior_question_elapsed_time','prior_question_had_explanation','bundle_id',\
                'part','tags1','tags2','tags3']
USE_COLS_PART = ['user_part_mean','user_part_sum','part_ans_mean']
USE_COLS_USER_ID = ['past_correctly_sum','past_not_correctly_sum','past_correctly_mean']
USE_COLS_CONTENT = ['content_id_ans_mean','content_id_num']
USE_COLS = USE_COLS_BASE + USE_COLS_PART + USE_COLS_USER_ID + USE_COLS_CONTENT
TARGET = 'answered_correctly'

CAT_FEATURES = ['part','tags1','tags2','tags3','content_id','task_container_id','prior_question_had_explanation','bundle_id']


def run_lgb(train,valid,LOG):
    lgb_params = {
                'n_estimators': 24000,
                'objective': 'binary',
                'boosting_type': 'gbdt',
                'metric': 'auc',
                'max_depth': 7,
                'learning_rate': 0.1,
                'seed': 127,
                'early_stopping_rounds': 50
            }
    train_x,valid_x = train[USE_COLS],valid[USE_COLS]
    train_y,valid_y = train[TARGET],valid[TARGET]
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



    # for n in range(1,6):
    #     train_index = pd.read_feather(f'./data/train_valid/cv{n}_train.feather')
    #     valid_index = pd.read_feather(f'./data/train_valid/cv{n}_train.feather')
    #     cv_train = train_df[train_df['row_id'].isin(train_index['row_id'])]
    #     cv_valid = train_df[train_df['row_id'].isin(valid_index['row_id'])]
    #     cv_train_x,cv_valid_x = cv_train[USE_COLS],cv_valid[USE_COLS]
    #     cv_train_y,cv_valid_y = cv_train[TARGET],cv_valid[TARGET]

    #     lgb_train = lgb.Dataset(cv_train_x, cv_valid_x)
    #     lgb_eval = lgb.Dataset(cv_train_y, cv_valid_y)

    #     del cv_train_x,cv_valid_x
    #     gc.collect()

    #     LOG.info(f'cv{n} start lgb train')
    #     t0 = time.time()
    #     model = lgb.train(lgb_params, lgb_train, valid_sets=[lgb_train, lgb_eval], verbose_eval=10,categorical_feature=CAT_FEATURES)
    #     LOG.info(f'cv{n} end lgb train : {time.time() - t0} s')

    #     val_pred = model.predict(cv_valid_x)
    #     score = roc_auc_score(cv_valid_y, val_pred)
    #     LOG.info(f"cv{n} AUC = {score}")
    #     models.append(model)

    #     fi[f'cv{n}_importance'] = model.feature_importance(importance_type="gain")
    #     oof += val_pred.tolist()
    #     ans += cv_valid_y.tolist()

    #     del model
    #     gc.collect()


    return model,fi,val_pred


def main():

    file_name = os.path.basename(__file__)[:-3]
    LOG = logger.Logger(name=f'{file_name}',filename=file_name)
    LOG.info('base line')
    LOG.info(f'{USE_COLS}')

    train_df = data_util.load_features(feature_list)
    train_index = pd.read_feather(f'./data/train_valid/cv1_train.feather')
    valid_index = pd.read_feather(f'./data/train_valid/cv1_valid.feather')
    train = train_df[train_df['row_id'].isin(train_index['row_id'])]
    valid = train_df[train_df['row_id'].isin(valid_index['row_id'])]

    del train_df
    gc.collect()

    LOG.info(f'train_size:{train.shape} valid_size:{valid.shape}')

    model,fi,valid_pred = run_lgb(train=train,valid=valid,LOG=LOG)
    data_util.seve_model(model,fi,file_name)

    # oof.to_pickle(f'./data/oof/{file_name}.pkl')






if __name__ == "__main__":
    main()