from scipy.spatial.transform import Rotation
import converter_functions as cf
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import random
import pprint
import pickle
import copy
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_txt_path',default='',type=str)
    parser.add_argument('--dest_path',default='',type=str)
    parser.add_argument('--name_to_idx_dict',default='./src_preprocessing/embryo_IDs_as_ints_final.pkl',type=str)
    #parser.add_argument('--micron_scalars',default='1,1,1',type=str)
    parser.add_argument('--micron_scalars',default='0.116,0.116,0.122',type=str)
    parser.add_argument('--zero_center',default=1,type=int)
    parser.add_argument('--per_emb_format',default=0,type=int, help='is entire time series in single file (as in Zhuo format)?')
    #parser.add_argument('--want_train',default=0,type=int)
    parser.add_argument('--want_train',default=0,type=int)
    parser.add_argument('--want_inf',default=1,type=int)
    #parser.add_argument('--num_rotations',default=0,type=int)
    parser.add_argument('--num_rotations',default=0,type=int)
    # parser.add_argument('--want_canonical',default=0,type=int)
    parser.add_argument('--want_canonical',default=1,type=int)
    parser.add_argument('--range_normalize',default=0,type=int)
    #parser.add_argument('--',default='',type=str)
    args = parser.parse_args()
    source_path = args.source_txt_path
    dest_path = args.dest_path
    zero_center = args.zero_center
    per_emb_format = args.per_emb_format
    want_train=args.want_train
    want_inf=args.want_inf
    num_rotations=args.num_rotations
    name_to_idx_dict=args.name_to_idx_dict
    want_canonical=args.want_canonical
    range_normalize=args.range_normalize

    # load central name-to-index file to assign int versions of names for training with npy data types
    with open(name_to_idx_dict,"rb") as f:
        name_to_index = pickle.load(f)
    for key in name_to_index:
        name_to_index[key] = name_to_index[key].strip()


    # load per-emb folder 
    for emb_fold in os.listdir(source_path):
        print("current emb:",emb_fold)
        emb_folder = os.path.join(source_path,emb_fold)

        if zero_center == 1:

            agg_for_zc = []
            for emb_f in os.listdir(emb_folder):
                emb_file = os.path.join(emb_folder,emb_f)
                df = pd.read_csv(emb_file,header=0)
                agg_for_zc.append(df)
            agg_df = pd.concat(agg_for_zc)
            if args.micron_scalars != '1,1,1':
                micron_scalars = [float(i) for i in args.micron_scalars.split(',')]
                agg_df = cf.micron_scale(df=agg_df,voxel_size=micron_scalars)
            center_agg_df,midpt = cf.zero_center(agg_df)

        for emb_f in os.listdir(emb_folder):
            emb_file = os.path.join(emb_folder,emb_f)
            comb_filename = emb_fold + '_' + emb_f.split('.')[0] + '_'
            #print("comb filename:",comb_filename)
            #print("current file:",emb_f)
            df = pd.read_csv(emb_file,header=0)
            #print("reading orig csv right?")
            #print(df['name'])
            stage = len(df)
            #print("stage of file:",stage)
            # do zero-center all together if not done ... build df TODO
            # first preprocess - micron scale
            if args.micron_scalars != '1,1,1':
                micron_scalars = [float(i) for i in args.micron_scalars.split(',')]
                df = cf.micron_scale(df=df,voxel_size=micron_scalars)
            if zero_center == 1:
                df, _ = cf.zero_center(df,midpoint=midpt) # midpoint calculated from prev. aggregation above
        

            if range_normalize == 1: # never used in any final things
                df = cf.range_normalize(df)

            per_orig_ptcloud_dict_list = []
            try:
                canonical_dict = cf.get_leifer_dict(df=df,name_to_idx=name_to_index,file=emb_f)
                #print("are int names assigned correctly (whitespace issue)?")
                #pprint.pprint(canonical_dict)
            except ValueError:
                #print("val error!")
                continue # this to skip point clouds with very early/late (out of scope) IDs

            if want_canonical == 1:
                # per_orig_ptcloud_df_list.append(canonical_dict)
                per_orig_ptcloud_dict_list.append(canonical_dict)

            for num_rots in range(num_rotations):
                leifer_dict_cpy = copy.deepcopy(canonical_dict)
                rotated_dict = cf.randomly_rotate(pt_cloud_dict=leifer_dict_cpy)
                per_orig_ptcloud_dict_list.append(rotated_dict)

            # just doing these fxns here b/c not really fxns...
            if want_train == 1:
                dest_train_main_path = dest_path + '/npy_format/'
                if not os.path.exists(dest_train_main_path):
                    #print(dest_train_main_path, 'doesn\'t exist!')
                    os.mkdir(dest_train_main_path)
                dest_train_stage_path = dest_train_main_path + 'worms'
                if not os.path.exists(dest_train_stage_path):
                    #print(dest_train_stage_path, 'doesn\'t exist!')
                    os.mkdir(dest_train_stage_path)
                i=0
                for rot_version in per_orig_ptcloud_dict_list:
                    if i==0 and want_canonical == 1:
                        train_filename = comb_filename + '_canonical.npy'
                    else:
                        train_filename = comb_filename + str(i) + '.npy'
                    dest_train_file_path = os.path.join(dest_train_stage_path,train_filename)
                    #print("dest_train_file_path:",dest_train_file_path)
                    pts = rot_version['pts'][:, :]

                    with open(dest_train_file_path, 'wb') as f:
                        np.save(f, pts)
                    i+=1


            if want_inf == 1:
                dest_inf_main_path = dest_path + '/pkl_format/'
                if not os.path.exists(dest_inf_main_path):
                    os.mkdir(dest_inf_main_path)
                dest_inf_stage_path = dest_inf_main_path + 'worms'
                if not os.path.exists(dest_inf_stage_path):
                    os.mkdir(dest_inf_stage_path)
                i=0
                for rot_version in per_orig_ptcloud_dict_list:
                    if i==0 and want_canonical == 1:
                        inf_filename = comb_filename + '_canonical.pkl'
                    else:
                        inf_filename = comb_filename + str(i) + '.pkl'
                    dest_inf_file_path = os.path.join(dest_inf_stage_path,inf_filename)
                    #print("dest_inf_file_path:",dest_inf_file_path)

                    with open(dest_inf_file_path,"wb") as f:
                        pickle.dump(rot_version,f)
                    i+=1
