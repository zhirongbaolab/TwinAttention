from prediction_path_arg import predict_label
import numpy as np
import statistics
import random
import pickle
import os

def calculate_similarity(list1, list2):
    if len(list1) != len(list2):
        print("Different stages being paired!")
        return 0.0

    count = 0
    for i in range(len(list1)):
        if list1[i] == list2[i]:
            count += 1

    similarity = (count / len(list1)) * 100
    return similarity

def calculate_shared_similarity(list1, list2):
    shared_cells = list(set(list1) & set(list2))
    if len(shared_cells) == 0:
        return 0.0
    #print("length of shared cells:",len(shared_cells))
    #print("shared cells:", shared_cells)

    count = 0
    for i in range(len(list1)):
        if list1[i] in shared_cells:
            if list1[i] == list2[i]:
                count += 1
  
    shared_similarity = (count / len(shared_cells)) *100
    return shared_similarity

def calculate_stable_similarity(list1, list2,stable_cells):
    count = 0
    for i in range(len(list1)):
        if list1[i].strip() in stable_cells:
            if list1[i] == list2[i]:
                count += 1

    stable_similarity = (count / len(stable_cells)) * 100
    return stable_similarity

def calculate_unstable_similarity(list1, list2,stable_cells):
    count = 0
    for i in range(len(list1)):
        if list1[i].strip() not in stable_cells:
            if list1[i] == list2[i]:
                count += 1
    unstable_similarity = (count / (len(list1)-len(stable_cells))) * 100
    return unstable_similarity


def calculate_accuracy_L1():

    folder_path='./data/data_final/test/pkl_format/worms'
    folder_path_template = './data/data_final/template/pkl_format/worms'

    if not os.path.exists(folder_path):
        return None, None
    
    if not os.path.exists(folder_path_template):
        return None, None

    folder = os.listdir(folder_path)
    folder_template = os.listdir(folder_path_template)

    file_names = list(os.listdir(folder_path))
    file_names_template = list(os.listdir(folder_path_template))

    shared_accuracy_measures = []
    accuracy_measures=[]
    share_percents_measures = []

    number_pairs_to_form = len(file_names)
    num_template = len(file_names_template)

    for i in range(number_pairs_to_form):
        print("Processing pair number: ", i)
        print(file_names[i])
        # random.shuffle(file_names_template)
        # file1 = file_names_template[0]
        # file1 = os.path.join(folder_path_template, file1)

        # with open(file1,'rb') as f:
        #     temp = pickle.load(f)
        
        file2 = file_names[i]
        file2 = os.path.join(folder_path, file2)

        with open(file2,'rb') as f:
            test = pickle.load(f)

        len_test = len(test['name'])

        minimum_diff = 1000000
        template_name = 'None'

        for j in range(num_template):
            file1 = file_names_template[j]
            file1 = os.path.join(folder_path_template, file1)

            with open(file1,'rb') as f:
                temp_t = pickle.load(f)

            len_temp_t = len(temp_t['name'])

            if np.abs(len_temp_t - len_test) < minimum_diff:
                minimum_diff = np.abs(len_temp_t - len_test)
                temp = temp_t
                template_name = file1

        print("The template file is: ", template_name)

        #print("predicting alignment between ",file1, "and", file2)
        # assign args to components of data
        temp_pos = temp['pts']
        temp_label = temp['name']
        test_pos = test['pts']

        # not using color data
        temp_color=None
        test_color=None

        # perform prediction via Leifer method
        test_label, candidate_list = predict_label(temp_pos, temp_label, test_pos, temp_color,test_color, cuda=True)
        pred_label = [cell_name for cell_name, conf in test_label]
        truth_test_label = list(test['name'])

        accuracy_now = calculate_shared_similarity(pred_label, truth_test_label)
        print("The shared accuracy for this pair is: ",accuracy_now)

        accuracy_measures.append(accuracy_now)

        shared_cells = list(set(pred_label) & set(truth_test_label))

        # save the test pts and labels to a csv file
        file_name = './output/' + file2.split('/')[-1] + '_predicted.csv'
        with open(file_name, 'w') as f:
            for i in range(len(test_pos)):
                f.write(str(pred_label[i]) + '\n')
        
        


    avg_accuracy_measure=sum(accuracy_measures)/len(accuracy_measures)
    #avg_shared_cells_percentage = sum(share_percents_measures) / len(share_percents_measures)

    return avg_accuracy_measure


accuracy = calculate_accuracy_L1()
print("The average shared accuracy is: ",accuracy)



