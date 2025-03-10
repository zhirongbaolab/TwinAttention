import os

os.system('python ./src_preprocessing/main_convert.py --source_txt_path ./data/txt_file_by_emb/test --dest_path ./data/data_final/test --micron_scalars 0.254,0.254,1')
os.system('python ./src_preprocessing/main_convert.py --source_txt_path ./data/txt_file_by_emb/template --dest_path ./data/data_final/template --micron_scalars 0.254,0.254,1')
os.system('python ./pair_inference.py')
print("Finished! The results are stored in the 'output' folder.")