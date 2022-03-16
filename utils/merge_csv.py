import pandas as pd

# #MERGE NAME AND PATH
# df1 = pd.read_csv('../data/merav/2-names-bert.csv', sep=',',encoding="ISO-8859-8")
# print(df1.head())
# df2 = pd.read_csv('../data/merav/merav2-all.csv',encoding="ISO-8859-8")
# print(df2.head())
# new_df = df2.join(df1.set_index('path'), on="path",lsuffix='l')
# # new_df=pd.concat([df1.set_index("path"),df2.set_index('file_name')], axis=1, join='inner')
# # df2['label'] =df1['label']
# # df2['path'] = df1['path']
# print(new_df)
# new_df.to_csv("../data/merav/merav2-names.csv",encoding="ISO-8859-8")
# #
# # MERGE BERT WITH VECTORS - NO FT MODEL
# df1 = pd.read_csv('../data/merav3_names_bert.csv', sep=',',encoding="ISO-8859-8")
# print(df1.head())
# df2 = pd.read_csv('../data/last3_8_vec_emotion.csv', sep=',', encoding="ISO-8859-8")
# print(df2.head())
# new_df = df2.join(df1.set_index('path'), on="path",how="inner",lsuffix='l')
# print(new_df)
# new_df.to_csv('../data/last3_8_vec_emotion.csv',encoding="ISO-8859-8")
# #

# # # MERGE BERT WITH VECTORS - FT MODEL (5 fold)
for i in range(5):
    df1 = pd.read_csv('../data/cheat/cheat_bert_outputs_all.csv', sep=',')
    print(df1.head())
    df2 = pd.read_csv(f'../data/cheat/emotion_3_2_10_0.4/last_train{i}.csv', sep=',')
    print(df2.head())
    new_df = df2.join(df1.set_index('path'), on="path",how="inner",lsuffix='l')
    print(new_df)
    new_df.to_csv(f'../data/cheat/emotion_3_2_10_0.4/last_train{i}.csv',index=False)


# df1 = pd.read_csv('../data/cheat_updated/cheat_all.csv', sep=',')
# print(df1.head())
# df2 = pd.read_csv(f'C:/Users/noaai/Desktop/new_claims/final_claims/claim_with_asr_filtered_23_12.csv',
#                   sep=',',usecols=['file_name','ASR'])
# print(df2.head())
# new_df = df1.join(df2.set_index('file_name'), on="path",how="inner",lsuffix='l')
# print(new_df)
# new_df.to_csv(f'../data/cheat_updated/cheat_all.csv',index=False)


# merge wav2vec with fsfm features
# df1 = pd.read_csv('../data/cheat/last_base_without_ft.csv', sep=',',usecols=['path','vec','label'])
# print(df1.head())
# df2 = pd.read_csv('../data/cheat/cheat_with_fsfm.csv',
#                   sep=',',usecols=['file','fsfm'])
# print(df2.head())
# new_df = df1.join(df2.set_index('file'), on="path",how="inner",lsuffix='l')
# print(new_df)
# new_df.to_csv(f'../data/chat/cheat_names_with_fsfm.csv',index=False)
