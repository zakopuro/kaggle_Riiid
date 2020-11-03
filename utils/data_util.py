import pandas as pd
import pickle



def load_features(features_list):

    dfs = [pd.read_feather(f'./features/data/{feature}_train.feather') for feature in features_list]
    df = pd.concat(dfs,axis=1)
    return df



def seve_model(model,fi,file_name):
    with open(f'./models/{file_name}','wb') as f:
        pickle.dump(model,f)

    fi.to_csv(f'./features/fe/{file_name}.csv',index=False)