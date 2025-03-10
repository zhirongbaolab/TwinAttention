import os
import statistics
import numpy as np
import random
import argparse
import shutil

# def main(root_root_source_folder, root_batch_folder,per_stg_data_factor,batch_sz,num_embs, num_rots_in_set): # This line is commented out by Ruohan on 9 Oct 2024
def main(root_source_folder, root_batch_folder, per_stg_data_factor, batch_sz, num_embs, num_rots_in_set): # This line is added by Ruohan on 9 Oct 2024
    # Get a list of all subfolders in the root folder
    subfolders = [f for f in os.listdir(root_source_folder) if os.path.isdir(os.path.join(root_source_folder, f))]

    # Create a dictionary to store the samples for each subfolder
    samples_per_subfolder = {}

    for subfolder in subfolders:
        # Parse the X value from the subfolder name
        try:
            stg_num = int(subfolder.split('c')[0])
        except ValueError:
            continue

        samples_per_subfolder[subfolder] = {
            'samples':[],
            'n': 0,
            'm': 0
        }

        # Try to collect samples from this subfolder
        samples_to_collect = [os.path.join(root_source_folder,subfolder,f) for f in os.listdir(os.path.join(root_source_folder, subfolder))]
        #print("samples to collect:",samples_to_collect)
        #samples_per_subfolder[subfolder]['samples'].extend(samples_to_collect)

        num_samples_plateau = 3*num_embs*num_rots_in_set

        if len(samples_to_collect) < num_samples_plateau: # don't collect more if already in plateau
            if stg_num == 579: # 579 is just to show the running process of the script
                print('579: extending window - not plateau')
            #window_size=max((stg_num//5),5)
            if stg_num < 50:
                window_size=5
            elif stg_num >= 50 and stg_num < 100:
                window_size = 10
            elif stg_num >= 100 and stg_num < 200:
                window_size = 20
            elif stg_num >= 200 and stg_num < 300:
                window_size = 30
            elif stg_num >= 300 and stg_num < 400:
                window_size = 30
            elif stg_num >= 400 and stg_num < 500:
                window_size = 50
            elif stg_num >= 500:
                window_size = 50
            window_radius = window_size//2

            # Extend collection across window
            for j in range(1,window_radius+1):
                subfolder_prev = f"{stg_num - j}cells"
                if subfolder_prev in subfolders:
                    prev_sub_fold = os.path.join(root_source_folder,subfolder_prev)
                    if len(os.listdir(prev_sub_fold)) < num_samples_plateau: # also don't collect from adjacent plateau
                         samples_to_collect.extend([os.path.join(root_source_folder,prev_sub_fold,f) for f in os.listdir(prev_sub_fold)])
                    else: # break out upon hitting adjacent plateau
                        if stg_num == 579:
                            print("hit prev plateau:",subfolder_prev)
                        #print("hit prev plateau:",subfolder_prev)
                        break

            # Extend collection across window
            for j in range(1,window_radius+1):
                subfolder_next = f"{stg_num + j}cells"

                if subfolder_next in subfolders:
                    nxt_sub_fold = os.path.join(root_source_folder,subfolder_next)
                    if len(os.listdir(nxt_sub_fold)) < num_samples_plateau:
                        samples_to_collect.extend([os.path.join(root_source_folder,nxt_sub_fold,f) for f in os.listdir(nxt_sub_fold)])
                    else:
                        #print("hit next plateau:",subfolder_next)
                        break
        else:
            print("at a plateau!")
        samples_per_subfolder[subfolder]['samples'] = samples_to_collect
        samples_per_subfolder[subfolder]['n'] = len(os.listdir(os.path.join(root_source_folder,subfolder)))
        samples_per_subfolder[subfolder]['m'] = len(samples_to_collect)

    for stg in samples_per_subfolder:
        #print("curr stg:",stg)
        curr_dict=samples_per_subfolder[stg]
        num_samples=curr_dict['n']
        num_samples_window = curr_dict['m']
        #print("num samples:",num_samples)
        samples=curr_dict['samples']

        num_stage = int(stg.split('c')[0])

        sample_list=samples
        random.shuffle(sample_list)
        if num_samples > 0:
            # capping on!
            # here min (m, cap)
            amount_per_stage = min(num_samples_plateau,per_stg_data_factor*num_samples_window) # TESTING WITHOUT CAP 3 NOV #min(per_stg_data_factor*num_samples_window,num_samples_plateau)
            # alt cap - min(cn,m)
            # amount_per_stage = min(per_stage_data_factor*num_samples, num_samples_window)
        else:
            continue
        if num_stage == 579:
            print("579 amt per stg:",amount_per_stage)

        num_batches = int(amount_per_stage / batch_sz)
        #print("num batches:",num_batches)
        for i in range(num_batches):
            batch_folder = os.path.join(root_batch_folder, f"{num_stage}cell_batch{i + 1}")
            #print('making new train dir:',batch_folder)
            os.makedirs(batch_folder, exist_ok=True)
            if amount_per_stage <= len(sample_list):
                #print("amount per stg <= m! amount samples:",amount_per_stage,"m:",len(sample_list))
                start_idx=(i*batch_sz)
                batch_samples = sample_list[start_idx:(start_idx+batch_sz)]
                #print("len batch samples:",len(batch_samples))
                #print("batch samples:",batch_samples)
                for sample in batch_samples:

                    src_path=sample
                    just_sample_file = sample.split("/")[-1]

                    dst_path = os.path.join(batch_folder, just_sample_file)
                    #print("NOT COPYING BUT WOULD:", dst_path)
                    shutil.copy(src_path, dst_path)

            else:
                #print("sampling with replacement!! at stage:",stg)
                batch_samples = random.sample(samples, batch_sz)
                for sample in batch_samples:
                    just_sample_file=sample.split("/")[-1]
                    #source_subfolder=next(sf for sf in samples_per_subfolder if sample in samples_per_subfolder[sf])
                    #src_path = os.path.join(root_source_folder, source_subfolder, sample)
                    src_path=sample
                    dst_path = os.path.join(batch_folder, just_sample_file)
                    #print("NOT COPYING BUT WOULD WITH REPLACEMENT:",sample, dst_path)
                    shutil.copy(src_path, dst_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_source_folder',default='../data/real_data_Ruohan/data_sampling_final/train/npy_format/',type=str,help='source folder of staged folders containing npy files')
    parser.add_argument('--root_batch_folder',default='../data/real_data_Ruohan/data_sampling_final/train/batched/',type=str,help='destination dir of all batch folders')
    parser.add_argument('--per_stg_data_factor',default=1,type=int,help='depth of sampling, c')
    parser.add_argument('--num_embs',default=155,type=int,help='depth of sampling, c')
    parser.add_argument('--batch_sz',default=10,type=int,help='batch size, which here equals the target training folder sizes to control exact batch contents')
    parser.add_argument('--num_rots_in_set',default=11,type=int,help='number of rotations performed per point cloud, for use in calculation of true unique #s of datapoints in the set')
    args = parser.parse_args()
    root_source_folder=args.root_source_folder
    root_batch_folder=args.root_batch_folder
    per_stg_data_factor=args.per_stg_data_factor
    num_embs=args.num_embs
    batch_sz=args.batch_sz
    num_rots_in_set = args.num_rots_in_set

    if num_rots_in_set == 0:
        print('Please indicate number of rotations per point cloud in the set with --num_rots_in_set. Use help for the flag for more info')

    if num_embs == 0:
        print('Please indicate number of embryos in the set with --num_embs. Use help for the flag for more info')
    
    if batch_sz == 0:
        print('Please indicate the batch size, which here equals the target training folder sizes to control exact batch contents with --batch_sz. Use help for the flag for more info')
    
    #root_source_folder='/scratch/hause/13dec_pv_sim_v2/train/all_npys/'#'/scratch/hause/12dec_pv_noise_added_sims/eval/all_npys/'#'/bao_data_zrc/baolab/erinhaus/mean_centered/txt_format/split_for_training/npy_format/eval/by_stage'#'/bao_data_zrc/baolab/erinhaus/mean_centered/stage_centered_training/per_stage_all_npys_10rots/per_stage_npys/train'#/scratch/hause/29nov_redo_orig_sim/all_npys/eval'#'/bao_data_zrc/baolab/erinhaus/simulation_data_november/eval/noise_added/npy_format/'#'/bao_data_zrc/baolab/erinhaus/mean_centered/stage_centered_training/per_stage_all_npys_10rots/per_stage_npys/eval/'#'/bao_data_zrc/baolab/erinhaus/mean_centered/txt_format/split_for_training/npy_format/train/by_stage/'
    #root_source_folder = "/bao_data_zrc/baolab/erinhaus/mean_centered/stage_centered_training/per_stage_all_npys_10rots/per_stage_npys/eval/"
    #root_batch_folder = '/scratch/hause/13dec_pv_sim_v2/train/batched/'#'/scratch/hause/5dec_realdata_cmdepth_dbW_batched/eval'#/scratch/hause/1dec_orig_sim_2nov_depth_drbao_w/eval'#/bao_data_zrc/baolab/erinhaus/simulation_data_november/eval/noise_added/batched_nov2sampling_c1'#'/bao_data_zrc/baolab/erinhaus/mean_centered/stage_centered_training/per_stage_all_npys_10rots/batch_schemes/6nov_sim_allstg_nocap/train'#root_batch_folder='/bao_data_zrc/baolab/erinhaus/mean_centered/stage_centered_training/per_stage_all_npys_10rots/batch_schemes/5nov_sim_2novsamplestrat/train'#'/bao_data_zrc/baolab/erinhaus/mean_centered/stage_centered_training/per_stage_all_npys_10rots/batch_schemes/2nov_c1_3cutoff/eval'#'/bao_data_zrc/baolab/erinhaus/mean_centered/stage_centered_training/per_stage_all_npys_10rots/batch_schemes/13oct_adaptive_c2w10/train'
    #per_stg_data_factor=1
    #num_embs = 99 # CHANGE FOR EVAL - can't scrape from per-stage folders
    #batch_sz=10
    #num_rots_in_set=10
    main(root_source_folder, root_batch_folder,per_stg_data_factor,batch_sz, num_embs, num_rots_in_set)

