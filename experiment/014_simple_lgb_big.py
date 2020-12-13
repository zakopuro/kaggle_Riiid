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

FEATURES_LIST = ['BASE','USER_ID','PART','USER_PART','CONTENT','USER_CONTENT','TAGS','USER_BUNDLE','USER_TAGS','USER_READING_PART','TYPE_OF','USER_TYPE_OF','CATEGORICAL']

USE_COLS = ['answered_correctly_avg_user','prior_question_elapsed_time','prior_question_had_explanation','bundle_id','part','tags1','tags2','tags3','l_type_of','content_id',\
            'answered_correctly_sum_user','task_container_id','count_user','answered_correctly_avg_part','answered_correctly_avg_user_part','answered_correctly_sum_user_part','count_user_part','answered_correctly_avg_content',\
            'answered_correctly_sum_content','content_num','answered_correctly_avg_tags1',\
            'answered_correctly_avg_user_tags1','answered_correctly_sum_user_tags1','count_user_tags1',\
            'answered_correctly_avg_user_reading_part','answered_correctly_sum_user_reading_part','count_user_reading_part','reading_part','answered_correctly_avg_reading_part',\
            'answered_correctly_avg_type_of','answered_correctly_avg_user_type_of','answered_correctly_sum_user_type_of','count_user_type_of',
            'user_id_part','user_id_content','user_id_bundle','user_id_tags1','user_id_reading_part','user_id_type_of']

TARGET = 'answered_correctly'

CAT_FEATURES = ['part','tags1','tags2','tags3','content_id','task_container_id','prior_question_had_explanation',\
                'bundle_id','user_id_part','user_id_content','user_id_bundle','user_id_tags1','user_id_reading_part','user_id_type_of']


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

    train = data_util.load_features(FEATURES_LIST,path='features/data',train_valid='train')
    valid = data_util.load_features(FEATURES_LIST,path='features/data',train_valid='valid')

    train = data_util.reduce_mem_usage(train)
    valid = data_util.reduce_mem_usage(valid)

    LOG.info(f'train_size:{train.shape} valid_size:{valid.shape}')

    model,fi,valid_pred = run_lgb(train=train,valid=valid,LOG=LOG)
    data_util.seve_model(model,fi,file_name)





if __name__ == "__main__":
    main()