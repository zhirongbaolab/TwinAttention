# imports
# from DNC_pred_get_embeddings import predict_label # This line is commented out by Ruohan on 9 Oct 2024
from src.model import predict_label # This line is added by Ruohan on 9 Oct 2024
import numpy as np
import pandas as pd
import statistics
import random
import pickle
import copy
import os
import torch
from concurrent.futures import ProcessPoolExecutor
import pyarrow
import fastparquet



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



# Define columns and initialize dataframes for storing results
columns = ['stage'] + ['cell_ID'] + [f'dim_{i+1}' for i in range(128)]
df_space = pd.DataFrame(columns=columns)
df_test = pd.DataFrame(columns=columns)

# Function to process a single stage
def process_stage(stage):
    print(f'Processing stage {stage}...')
    folder_path_test = f'../data/real_data_Ruohan/data_sampling_final/test/pkl_format/{stage}cells'
    folder_path_train = f'../data/real_data_Ruohan/data_sampling_final/train/pkl_format/{stage}cells'
    folder_path_eval = f'../data/real_data_Ruohan/data_sampling_final/eval/pkl_format/{stage}cells'

    # Skip this stage if any folder does not exist
    if not all(os.path.exists(folder) for folder in [folder_path_test, folder_path_train, folder_path_eval]):
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)

    # Load reference worm data
    file_names_train = os.listdir(folder_path_train)
    file_names_train = [file for file in file_names_train if 'canonical' in file]
    random.shuffle(file_names_train)
    with open(os.path.join(folder_path_train, file_names_train[0]), 'rb') as f:
        temp_train = pickle.load(f)

    # Create dataframes to store embeddings for this stage
    df_space_stage = pd.DataFrame(columns=columns)
    df_test_stage = pd.DataFrame(columns=columns)

    file_names_eval = os.listdir(folder_path_eval)
    file_names_eval = [file for file in file_names_eval if 'canonical' in file]
    file_names_test = os.listdir(folder_path_test)
    file_names_test = [file for file in file_names_test if 'canonical' in file]


    # Process train, eval, and test sets
    for file_names, folder_path, df_out in [(file_names_train, folder_path_train, df_space_stage),
                                            (file_names_eval, folder_path_eval, df_space_stage),
                                            (file_names_test, folder_path_test, df_test_stage)]:
        random.shuffle(file_names)
        for file_name in file_names:
            with open(os.path.join(folder_path, file_name), 'rb') as f:
                data = pickle.load(f)
            
            temp_pos = temp_train['pts']
            
            temp_label = temp_train['name']
            
            test_pos = data['pts']
            
            encodings = predict_label(temp_pos, temp_label, test_pos, None, None, cuda=True)
            test_encoding = encodings[0]['mov_emb'][1,:,:]
            
            # Add encoded data to dataframe
            for k in range(len(test_encoding)):
                df_out.loc[len(df_out)] = [stage] + [data['name'][k]] + test_encoding[k].tolist()

    return df_space_stage, df_test_stage

# Run each stage in parallel
with ProcessPoolExecutor(max_workers=15) as executor:
    results = list(executor.map(process_stage, range(2, 521)))

# Combine results
for space, test in results:
    df_space = pd.concat([df_space, space], ignore_index=True)
    df_test = pd.concat([df_test, test], ignore_index=True)

# Save dataframes to parquet files
df_space.to_parquet('metaPC_embeddings_new.parquet')
df_test.to_parquet('test_embeddings_new.parquet')
print('Done!')