# configuration
import os
HOME = r'F:\my-home\0-北交课程\02-深度学习\1-oneflow赛题\work_space_2\03_oneflow_challenge_vit'

# Training settings
batch_size = 64
epochs = 20
lr = 3e-5
gamma = 0.7
seed = 42

'''

    Traffic Signs Classify

'''
TS = {
    'num_classes': 10,  # 类别数
    'batch_size': 32,
    'epochs': 20,
    'lr': 3e-5,  # 初始学习率
    'lr_steps': (20, 40, 80),  # 学习率更新步长
    'gamma': 0.7,
    'seed': 42,
    'min_dim': 300,
    'max_iter': 100,  # 迭代次数
    'steps': [8, 16, 32, 64, 100, 300],
    'name': 'VOC',

}
