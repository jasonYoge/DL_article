import glob
import os
import pandas as pd  # 处理csv文件
import numpy as np

dir_name = 'data'
superpixels_name = 'ISIC_*_superpixels.png'
csv_name = 'ISIC-2017_Training_Part3_GroundTruth.csv'
data_list = {}
data_type = ['Melanoma', 'Nevus', 'seborrheic_keratosis']
validation_percentage = 10
test_percentage = 10


def delete_superpixels_from_data():
    dir_path = os.path.join(os.curdir, dir_name)
    file_path = os.path.join(dir_path, superpixels_name)
    if os.path.exists(dir_path):
        file_list = glob.glob(file_path)
        for path in file_list:
            os.remove(path)
        print('Removed superpixel files from data directory.')
    else:
        print("Directory name is wrong.")


def create_data_list():
    # 初始化图像数据字典
    for type_name in data_type:
        data_list[type_name] = {
            'images': [],
            'validation': [],
            'test': []
        }

    # 获取ground_truth数据的路径
    dir_path = os.path.join(os.curdir, dir_name)
    file_path = os.path.join(dir_path, csv_name)

    # 构造数据字典
    if os.path.exists(file_path):
        csv_info = pd.read_csv(file_path)
        for index, row in csv_info.iterrows():
            chance = np.random.randint(100)
            file_name = os.path.join(dir_path, row[0] + '.jpg')
            # 验证集
            if chance < validation_percentage:
                if row[1] == 1.0:
                    data_list[data_type[0]]['validation'].append(file_name)
                elif row[2] == 1.0:
                    data_list[data_type[2]]['validation'].append(file_name)
                else:
                    data_list[data_type[1]]['validation'].append(file_name)
            # 测试集
            elif chance < (test_percentage + validation_percentage):
                if row[1] == 1.0:
                    data_list[data_type[0]]['test'].append(file_name)
                elif row[2] == 1.0:
                    data_list[data_type[2]]['test'].append(file_name)
                else:
                    data_list[data_type[1]]['test'].append(file_name)
            # 训练集
            else:
                if row[1] == 1.0:
                    data_list[data_type[0]]['images'].append(file_name)
                elif row[2] == 1.0:
                    data_list[data_type[2]]['images'].append(file_name)
                else:
                    data_list[data_type[1]]['images'].append(file_name)
        print('Create data dict success.')
    else:
        print("CSV file path is wrong.")

if __name__ == '__main__':
    delete_superpixels_from_data()
    create_data_list()
