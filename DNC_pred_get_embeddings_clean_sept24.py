from scipy.optimize import linear_sum_assignment
import torch
import pickle
import numpy as np
# from get_encodings_model import NIT_Registration # This line is commented out by Ruohan on 9 Oct 2024
from src.model import NIT_Registration # This line is added by Ruohan on 9 Oct 2024
from scipy.special import softmax

#import matplotlib.pyplot as plt

# EDITED BY ERIN cuda= True orig!
def predict_label(temp_pos, temp_label, test_pos, temp_color=None, test_color=None, cuda=False, topn=5):
    model = NIT_Registration(input_dim=3, n_hidden=128, n_layer=6, p_rotate=1, feat_trans=1, cuda=cuda)
    device = torch.device("cuda" if cuda else "cpu")
    # load trained model - change to relevant model file
    model_path = './model/late_embryo_final.bin'
    #model_path = '/home/rrh/digit_life/embryo_extended/model/real_retrained_model/FINAL_MODEL_FOR_THESIS.bin'
    params = torch.load(model_path, map_location=lambda storage, loc: storage)

    # all_model_state_parameters = params['state_dict']
    # other_reqd_keys=["fc_outlier.weight", "fc_outlier.bias", "point_f.conv1.weight", "point_f.conv1.bias", "point_f.bn1.weight", "point_f.bn1.bias", "point_f.bn1.running_mean", "point_f.bn1.running_var", "enc_l.self_attn.in_proj_weight", "enc_l.self_attn.in_proj_bias", "enc_l.self_attn.out_proj.weight", "enc_l.self_attn.out_proj.bias", "enc_l.linear1.weight", "enc_l.linear1.bias", "enc_l.linear2.weight", "enc_l.linear2.bias", "enc_l.norm1.weight", "enc_l.norm1.bias", "enc_l.norm2.weight", "enc_l.norm2.bias"]
    # just_model_state_params=dict((k,all_model_state_parameters[k]) for k in params['state_dict'].keys() if "model." in k or k in other_reqd_keys)
    model.load_state_dict(params['state_dict'])  # added by ruohan 25 Oct

    model = model.to(device)

    # put template worm data and test worm data into a batch
    pt_batch = list()
    color_batch = list()

    pt_batch.append(temp_pos[:, :3])

    pt_batch.append(test_pos[:, :3])
    #pt_batch.append(test_pos[i][:, :3] for i in range(len(test_pos))) # here we can add more test worm if provided as a list.

    if temp_color is not None and test_color is not None:
        color_batch.append(temp_color)
        color_batch.append(test_color)
    else:
        color_batch = None
    data_batch = dict()
    data_batch['pt_batch'] = pt_batch
    data_batch['color'] = color_batch
    data_batch['match_dict'] = None
    data_batch['ref_i'] = 0

    model.eval()
    pt_batch = data_batch['pt_batch']
    with torch.no_grad():
        encoded_points = model(pt_batch,match_dict=None, ref_idx=data_batch['ref_i'], mode='eval')
       

    return encoded_points # dummy return by ERIN 24Aug
