# imports
# from DNC_pred_get_embeddings import predict_label # This line is commented out by Ruohan on 9 Oct 2024
from DNC_pred_get_embeddings_clean_sept24  import predict_label # This line is added by Ruohan on 9 Oct 2024
import numpy as np
import pandas as pd
import statistics
import random
import pickle
import copy
import os
import torch



# # load reference worm data
#   # to build a consistent embedding space, we used a single, randomly
#   # - selected worm to use as a universal reference via a worm in the train set


# # load test worm data


# # set temp_label as numeric reference labels
# # set temp_pos and test_pos as arrays of xyz points in the worm point clouds

# temp_color=None
# test_color=None

# # perform prediction with model
# encodings = predict_label(temp_pos, temp_label, test_pos, temp_color,test_color, cuda=True)

# # retrieve test encoding
# test_encoding = encodings[0]['mov_emb'][1,:,:]

# # add to a data structure (I have been saving as 128 columns of a Pandas dataframe)

# # add metadata back (I add cell ID, source embryo, etc., you could also add xyz and etc. for easier data analysis, or just merge with 3d positional file)

# # save to a file per-embryo or dataset (I use parquet), whatever you think is best for your prefered analysis format

# construct a dataframe to store the 128 embeddings, cell ID, source embryo and stage
columns = ['stage'] + ['cell_ID'] + [f'dim_{i+1}' for i in range(128)]
df_space = pd.DataFrame(columns=columns)
df_test = pd.DataFrame(columns=columns)

for i in range(2,551):
    print("Processing stage: ", i)
    stage = i
    folder_path_test = '../data/real_data_Ruohan/data_sampling_final/test/pkl_format/500cells'
    folder_path_test = folder_path_test.replace('500', str(stage))

    folder_path_train = '../data/real_data_Ruohan/data_sampling_final/train/pkl_format/500cells'
    folder_path_train = folder_path_train.replace('500', str(stage))

    folder_path_eval = '../data/real_data_Ruohan/data_sampling_final/eval/pkl_format/500cells'
    folder_path_eval = folder_path_eval.replace('500', str(stage))


    if not os.path.exists(folder_path_test):
        continue
    if not os.path.exists(folder_path_train):
        continue
    if not os.path.exists(folder_path_eval):
        continue

    # load reference worm data
    # to build a consistent embedding space, we used a single, randomly selected worm to use as a universal reference via a worm in the train set
    file_names_train = list(os.listdir(folder_path_train))
    random.shuffle(file_names_train)
    file1_train = file_names_train[0]
    file1_train = os.path.join(folder_path_train, file1_train)
    with open(file1_train, 'rb') as f:
        temp_train = pickle.load(f)
    
    file_names_eval = list(os.listdir(folder_path_eval))
    random.shuffle(file_names_eval)

    file_names_test = list(os.listdir(folder_path_test))
    random.shuffle(file_names_test)

    for j in range(len(file_names_train)):
        file2_train = file_names_train[j]
        file2_train = os.path.join(folder_path_train, file2_train)
        with open(file2_train, 'rb') as f:
            test_train = pickle.load(f)
        temp_pos = temp_train['pts']
        temp_label = temp_train['name']
        test_pos = test_train['pts']
        temp_color = None
        test_color = None
        encodings = predict_label(temp_pos, temp_label, test_pos, temp_color,test_color, cuda=True)
        test_encoding = encodings[0]['mov_emb'][1,:,:]
        for k in range(len(test_encoding)):
            df_space.loc[len(df_space)] = [stage] + [test_train['name'][k]] + test_encoding[k].tolist()

    for j in range(len(file_names_eval)):
        file2_eval = file_names_eval[j]
        file2_eval = os.path.join(folder_path_eval, file2_eval)
        with open(file2_eval, 'rb') as f:
            test_eval = pickle.load(f)
        temp_pos = temp_train['pts']
        temp_label = temp_train['name']
        test_pos = test_eval['pts']
        temp_color = None
        test_color = None
        encodings = predict_label(temp_pos, temp_label, test_pos, temp_color,test_color, cuda=True)
        test_encoding = encodings[0]['mov_emb'][1,:,:]
        for k in range(len(test_encoding)):
            df_space.loc[len(df_space)] = [stage] + [test_eval['name'][k]] + test_encoding[k].tolist()
    
    for j in range(len(file_names_test)):
        file2_test = file_names_test[j]
        file2_test = os.path.join(folder_path_test, file2_test)
        with open(file2_test, 'rb') as f:
            test_test = pickle.load(f)
        temp_pos = temp_train['pts']
        temp_label = temp_train['name']
        test_pos = test_test['pts']
        temp_color = None
        test_color = None
        encodings = predict_label(temp_pos, temp_label, test_pos, temp_color,test_color, cuda=True)
        test_encoding = encodings[0]['mov_emb'][1,:,:]
        for k in range(len(test_encoding)):
            df_test.loc[len(df_test)] = [stage] + [test_test['name'][k]] + test_encoding[k].tolist()
          
# put the dataframes into a parquet file
df_space.to_parquet('metaPC_embeddings.parquet')
df_test.to_parquet('test_embeddings.parquet')






