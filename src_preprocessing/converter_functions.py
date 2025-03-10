from scipy.spatial.transform import Rotation
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import random
import pprint
import pickle
import copy
import os


def micron_scale(df,voxel_size):
    df['x'] = df['x'].apply(lambda x: x*voxel_size[0])
    df['y'] = df['y'].apply(lambda x: x*voxel_size[1])
    df['z'] = df['z'].apply(lambda x: x*voxel_size[2])
    return df

def zero_center(df,midpoint=None):
    centered_df = df.copy()
    # if midpoint != None: # This line is commented out by Ruohan on 12 Oct 2024
    if midpoint is None: # This line is added by Ruohan on 12 Oct 2024
        midpoint = centered_df[['x','y','z']].mean()

    centered_df['x'] -= midpoint['x']
    centered_df['y'] -= midpoint['y']
    centered_df['z'] -= midpoint['z']

    return centered_df, midpoint

def range_normalize(df):
    normalized_df = df.copy()
    max_x = max(df['x'])
    max_y = max(df['y'])
    max_z = max(df['z'])

    normalized_df['x'] = normalized_df['x']/max_x
    normalized_df['y'] = normalized_df['y']/max_y
    normalized_df['z'] = normalized_df['z']/max_z

    return normalized_df

def get_leifer_dict(df,name_to_idx,file):

    if df['name'].apply(isinstance, args=(str,)).all():
        real_names = df['name'].str.strip().tolist()

        # assigning numerical names to use np data types for model purposes
        index_to_label = []
        #print("name to idx:")
        #pprint.pprint(name_to_idx.values())
        for int_name in real_names:
            if int_name.strip() in [x.strip() for x in name_to_idx.values()]:
                index_to_label.append(list(name_to_idx.keys())[list(name_to_idx.values()).index(int_name)])

        df['name_idx'] = index_to_label

    else:
        df['name'] = 'TBD'
        index_to_label = []
        for i in range(len(df)):
            index_to_label.append(-1)
        df['name_idx'] = index_to_label

    leifer_dict = {
        'color': None,
        'f_name' : file,
        'label' : np.asarray(index_to_label),
        'name': np.asarray(df['name'].str.strip().tolist()),
        'pts': np.asarray(df[['x','y','z','name_idx']].values.tolist()),
        'rot': None
         }

    return leifer_dict

def randomly_rotate(pt_cloud_dict):
    #print('pt cloud dict before rot:',pt_cloud_dict)
    # Perform random rotation
    rand_rotation = Rotation.random()
    pts_to_rotate=pt_cloud_dict['pts'][:,0:3].copy()

    rotated_pt = rand_rotation.apply(pts_to_rotate)
    pt_cloud_dict['pts'][:,0:3] = rotated_pt
    pt_cloud_dict['rot'] = rand_rotation
    #print('pt cloud dict after rot:',pt_cloud_dict)
    return pt_cloud_dict
