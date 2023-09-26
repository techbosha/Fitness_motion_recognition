import os 
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

#抓csv_dataset_1下的csv檔，然後轉成一個datarame，columns是label和csv_path
def get_filename_number_1(s):
    return int(s)
def get_filename_number_2(s): #得到mp4檔名稱裡面的數字，做為排序使用
    return int(s.split('_')[-1].split('.')[0])


def get_csvdata(csv_file_path):
    csv_path = []
    dir_files = os.listdir(csv_file_path)
    sorted_dir = sorted(dir_files, key = get_filename_number_1)
    for dirs in sorted_dir:
        file_path = os.path.join(csv_file_path, dirs)
        files = os.listdir(file_path)
        sorted_files = sorted(files, key = get_filename_number_2)
        for items in sorted_files:
            file_vedio_path = os.path.join(file_path, items) 
            csv_path.append((dirs, file_vedio_path))
        dir_label_df = pd.DataFrame(csv_path, columns = ['label', 'csv_path'])
        dir_label_df['label'] = dir_label_df['label'].apply(lambda x:int(x))
    return dir_label_df


def load_csv_from_csvpath(dir_label_df):
    """_summary_

    Args:
        dir_label_df (_type_): _description_

    Returns:
        _type_: tuple
    """
    num_samples = 10
    frame_length = 50
    X = []
    Y = []
    for item_label, item_csvpath in dir_label_df.values:
        data = pd.read_csv(item_csvpath)
        data_label = item_label
        
        if len(data) < frame_length:
            continue
        else:
            
            data_1 = data.iloc[ : int(len(data)*0.3) ] #data的前30%
            selected_indices_1 = np.random.choice(data_1.index, size = num_samples, replace = False)
            selected_data_1 = data_1.loc[selected_indices_1]
            sorted_data_1 = selected_data_1.sort_index()
            
            data_2 = data.iloc[int(len(data) * 0.3) + 1 : int(len(data) * 0.7)]
            selected_indices_2 = np.random.choice(data_2.index, size = num_samples, replace = False)
            selected_data_2 = data_2.loc[selected_indices_2]
            sorted_data_2 = selected_data_2.sort_index()
        
            data_3 = data.iloc[-int(len(data) * 0.3) : ]
            selected_indices_3 = np.random.choice(data_3.index, size = num_samples, replace = False)
            selected_data_3 = data_3.loc[selected_indices_3]
            sorted_data_3 = selected_data_3.sort_index()
            
            data_final = pd.concat([sorted_data_1, sorted_data_2, sorted_data_3], ignore_index = True)
        
            
        #轉換為numpy array 輸出
        numpy_data = data_final.to_numpy()
        numpy_label_data = np.int64(data_label)
        X.append(numpy_data)
        Y.append(numpy_label_data)
        ndarray_data = np.array(X) #ndarray_data (3,30,99) ##3個csv,30個row,99個欄位
        ndarray_label_data = np.array(Y)
        # ndarray_label_data = ndarray_label_data.reshape(len(Y),1)
        encoder = OneHotEncoder(categories = [range(13)], sparse = False)
        ndarray_label_data_onehot = encoder.fit_transform(ndarray_label_data.reshape(-1, 1))
    return ndarray_label_data_onehot, ndarray_data