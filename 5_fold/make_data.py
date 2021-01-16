#!/usr/bin/python3
# -*- coding: utf-8 -*-

# 导入CSV安装包
import csv
import random

train_list = []
# 1. 创建文件对象
csv_file = csv.reader(open('../train.csv','r'))
for train_data in csv_file:
    train_list.append(train_data)
random.shuffle(train_list)  # 打乱数据列表

def get_kfold_data(k, i, train_list):
    # 返回第 i+1 折 (i = 0 -> k-1) 交叉验证时所需要的训练和验证数据，X_train为训练集，X_valid为验证集
    fold_size = len(train_list) // k  # 每份的个数:数据总条数/折数（组数）

    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        data_valid = train_list[val_start:val_end]
        data_train = train_list[0:val_start] + train_list[val_end:]
    else:  # 若是最后一折交叉验证
        data_valid = train_list[val_start:]  # 若不能整除，将多的case放在最后一折里
        data_train = train_list[0:val_start]

    return data_train, data_valid

for i in range(5):
    data_train, data_valid = get_kfold_data(5,i,train_list)

    f = open('train_'+str(i)+'.csv','w',encoding='utf-8')
    csv_writer = csv.writer(f)
    for data in data_train:
        csv_writer.writerow(data)
    f.close()

    f = open('val_' + str(i) + '.csv', 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    for data in data_valid:
        csv_writer.writerow(data)
    f.close()