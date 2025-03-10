from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
from matplotlib.colors import to_rgba
from sklearn.neighbors import KDTree
from collections import OrderedDict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import pprint
import argparse

# Load the embeddings
metaPC_embed_df = pd.read_parquet('metaPC_embeddings_new.parquet', engine='pyarrow')
test_embed_df = pd.read_parquet('test_embeddings_new.parquet', engine='pyarrow')


y_train = metaPC_embed_df['cell_ID'].reset_index(drop=True)
#print(y_train)
y_test = test_embed_df['cell_ID'].reset_index(drop=True)
#print(y_test)



metaPC_embed_df = metaPC_embed_df.reset_index(drop=True)
test_embed_df = test_embed_df.reset_index(drop=True)

X_train = metaPC_embed_df.drop(columns=['stage', 'cell_ID']).reset_index(drop=True)
#print(X_train)
X_test = test_embed_df.drop(columns=['stage', 'cell_ID']).reset_index(drop=True)
#print(X_test)
# Train a KNN classifier
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
#print("y_pred: ", y_pred)

print("Done predicting")

def get_accuracy(y_pred,X_test_lbl):
    X_test_lbl['prediction'] = y_pred
    #cols_to_compare=['cell_ID','prediction']

    # per-stage, get count of preds that match label
    #precision_per_stage=pd.DataFrame(columns=['stage','precision'])
    accs_per_stg=[]
    group_per_stage = X_test_lbl.groupby('stage')
    #print(group_per_stage)
    stages=[]
    pts_per_stg_list=[]
    accs_per_cell_per_stage=[]
    overall_acc=len(X_test_lbl.loc[X_test_lbl['cell_ID'] == X_test_lbl['prediction']])/len(X_test_lbl)
    for stg, group in group_per_stage:
        #stgs=[]
        num_pts_stg = len(group)/float(stg)
        pts_per_stg_list.append(num_pts_stg)
        #stages.append(stg)
        #val_count_cell = group['cell_ID'].value_counts()
        #cells_at_stg = val_count_cell.index.tolist()
        #print("group:",group)
        acc_for_stage=len(group.loc[group['cell_ID'] == group['prediction']])/len(group)

        print(len(group.loc[group['cell_ID'] == group['prediction']])/len(group))
        stages.append(stg)

        accs_per_stg.append(acc_for_stage)
        #print(accs_per_stg)
    return stages, accs_per_stg, overall_acc, pts_per_stg_list

stages, accs_per_stg, overall_acc, pts_per_stg_list = get_accuracy(y_pred,test_embed_df)
print("Overall accuracy: ", overall_acc)

# Save the results
results = pd.DataFrame({'stage': stages, 'accuracy': accs_per_stg, 'num_pts': pts_per_stg_list})
results.to_csv('knn_results_k_30.csv', index=False)
print('Done!')
