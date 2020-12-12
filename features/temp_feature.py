import pickle
import pandas as pd


with open('./features/kernel_data/user_tags_dict_sum.pkl','rb') as f:
    bundle_sum = pickle.load(f)

with open('./features/kernel_data/user_tasg_dict_count.pkl','rb') as f:
    bundle_count = pickle.load(f)

bundle_sum_key = list(bundle_sum.keys())
bundle_sum_values = list(bundle_sum.values())
bundle_count_key = list(bundle_count.keys())
bundle_count_values = list(bundle_count.values())
df_bundle_sum = pd.DataFrame()
df_bundle_sum['user_id_tags1'] = bundle_sum_key
df_bundle_sum['answered_correctly_sum_user_tags1'] = bundle_sum_values
df_bundle_count = pd.DataFrame()
df_bundle_count['user_id_tags1'] = bundle_count_key
df_bundle_count['count_user_tags1'] = bundle_count_values

df_bundle = pd.concat([df_bundle_sum.reset_index(drop=True), df_bundle_count.set_index('user_id_tags1').reindex(df_bundle_sum['user_id_tags1'].values).reset_index(drop=True)], axis=1)
df_bundle['answered_correctly_avg_user_tags1'] = df_bundle['answered_correctly_sum_user_tags1'] / df_bundle['count_user_tags1']

df_bundle.to_feather('./features/kernel_data/tags1_dict.feather')