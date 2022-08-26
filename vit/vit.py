import json

import torch
# from vit_pytorch import ViT
from vit_pytorch.efficient import ViT
# from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
# from tqdm.notebook import tqdm
from tqdm import tqdm
from config import HOME

# v = ViT(
#     image_size=224,  # 图片尺寸
#     patch_size=32,  # 批量大小
#     num_classes=10,  # 类别数目
#     dim=1024,  # Last dimension of output tensor after linear transformation.
#     depth=6,  # Number of Transformer blocks.
#     heads=16,  # Number of heads in Multi-head Attention layer.
#     mlp_dim=2048,  # Dimension of the MLP (FeedForward) layer.
#     channels=3,  # Number of image's channels.
#     dropout=0.1,  # Dropout rate.
#     emb_dropout=0.1  # Embedding dropout rate.
#
# )


# Training settings
batch_size = 64  # 64
epochs = 500  # 20
step = 50
lr = 1e-3  # 3e-5
gamma = 0.1  # 0.7
seed = 42

print("batch_size: {}".format(batch_size))
print("epochs: {}".format(epochs))
print("lr: {}".format(lr))
print("gamma: {}".format(gamma))
print("seed: {}".format(seed))

classes = []


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print("using {} device.".format(device))

if torch.cuda.is_available():
    if device.type.find('cuda') >= 0:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

image_path_root = 'image_data'
train_image_path = 'train_dataset'
test_image_path = 'test_dataset'

train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        # transforms.RandomResizedCrop(224),
        transforms.RandomGrayscale(p=0.2),  # 灰度
        transforms.RandomResizedCrop(size=224, scale=(0.08, 0.5), ratio=(3.0 / 4.0, 4.0 / 3.0)),  # 随机大小，长宽比裁剪
        transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 1)),  # 高斯模糊，高斯核大小7x7，标准差0.1,1
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),  # 透视
        transforms.RandomHorizontalFlip(p=0.5),  # 以0.5的概率垂直翻转
        transforms.RandomVerticalFlip(p=0.5),  # 以0.5的概率水平翻转
        transforms.RandomRotation(degrees=(0, 360), expand=True),  # 随机旋转，[0,360]随机角度，旋转后扩大图片保证信息完整，默认绕中心点旋转。
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),  # 降低亮度，降低对比度，降低饱和度
        transforms.ColorJitter(brightness=1.5, contrast=1.5, saturation=1.5),  # 增加亮度，增加对比度，增加饱和度
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.RandomErasing(p=0.4, scale=(0.02, 0.33), ratio=(0.5, 2.5), value=0, inplace=False),  # 随机遮挡

        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)
# test_dir = r'F:\my-home\0-北交课程\02-深度学习\1-oneflow赛题\work_space_2\03_oneflow_challenge_vit\test_dataset'
assert os.path.exists(os.path.join(HOME, image_path_root, train_image_path)), "{} path does not exist.".format(
    train_image_path)

train_dataset = datasets.ImageFolder(root=os.path.join(HOME, image_path_root, train_image_path),
                                     transform=train_transform)
# test_dataset = datasets.ImageFolder(root=os.path.join(HOME, test_image_path))
# train_list = glob.glob(os.path.join(train_image_path, '/**/*.jpg'), recursive=True)
# test_list = glob.glob(os.path.join(HOME, test_image_path, '*.jpg'))

print(f"Train Data: {len(train_dataset)}")
# print(f"Test Data: {len(test_list)}")


# train_image_list = train_dataset.class_to_idx
# cla_dict = dict((val, key) for key, val in train_image_list.items())
# # write dict into json file
# json_str = json.dumps(cla_dict, indent=4)
# with open('class_indices.json', 'w') as json_file:
#     json_file.write(json_str)

# batch_size = 8
# nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
# print('Using {} dataloader workers every process'.format(nw))
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           generator=torch.Generator(device=device))
# test_loader = torch.utils.data.DataLoader(test_dataset,
#                                            batch_size=batch_size, shuffle=True,
#                                            num_workers=nw)
##num_workers=nw

print(f"Batch Number: {len(train_loader)}")

# random_idx = np.random.randint(1, len(train_image_list), size=9)
# fig, axes = plt.subplots(3, 3, figsize=(16, 12))
#
# for idx, ax in enumerate(axes.ravel()):
#     img = Image.open(train_dataset[idx])
#     # ax.set_title(labels[idx])
#     ax.imshow(img)

efficient_transformer = Linformer(
    dim=128,
    seq_len=49 + 1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)

model = ViT(
    dim=128,
    image_size=224,
    patch_size=32,
    num_classes=10,
    transformer=efficient_transformer,
    channels=3,
).to(device)
# ).cuda()

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=step, gamma=gamma, last_epoch=-1, verbose=True)

train_epoch_loss = []
train_epoch_acc = []
batch_id_global = 1
train_batch_loss = []
train_batch_acc = []

for epoch_id, epoch in enumerate(range(epochs), start=1):
    epoch_loss = 0
    epoch_accuracy = 0

    for batch_id, (data, label) in enumerate(tqdm(train_loader), start=1):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

        train_batch_loss.append([batch_id_global, epoch_loss.item()])
        train_batch_acc.append([batch_id_global, epoch_accuracy.item()])
        batch_id_global += 1

        # print(epoch_accuracy,epoch_loss)
    # with torch.no_grad():
    #     epoch_val_accuracy = 0
    #     epoch_val_loss = 0
    #     for data, label in valid_loader:
    #         data = data.to(device)
    #         label = label.to(device)
    #
    #         val_output = model(data)
    #         val_loss = criterion(val_output, label)
    #
    #         acc = (val_output.argmax(dim=1) == label).float().mean()
    #         epoch_val_accuracy += acc / len(valid_loader)
    #         epoch_val_loss += val_loss / len(valid_loader)

    # print(
    #     f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    # )
    scheduler.step()
    train_epoch_loss.append([epoch_id, epoch_loss.item()])
    train_epoch_acc.append([epoch_id, epoch_accuracy.item()])
    print(
        f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - lr: {optimizer.state_dict()['param_groups'][0]['lr']}\n"
    )
    if epoch_id % 100 == 0:
        torch.save(model.state_dict(), 'log/v5/v5_vis_modle_{}.pth'.format(epoch_id))

torch.save(model.state_dict(), 'log/v5/v5_vis_modle_final.pth')

# show loss curve
plt.figure()
train_epoch_loss = np.array(train_epoch_loss)
train_epoch_acc = np.array(train_epoch_acc)
# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('epochs')  # x轴标签
plt.ylabel('loss')  # y轴标签

# 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
# 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
plt.plot(train_epoch_loss[:, :1], train_epoch_loss[:, 1:], linewidth=1, color='red', linestyle="solid",
         label="train epoch loss")
plt.plot(train_epoch_acc[:, :1], train_epoch_acc[:, 1:], linewidth=1, color='blue', linestyle="solid",
         label="train epoch acc")
plt.legend()
plt.title('Train Epochs Loss Curve')
# plt.show()
plt.savefig('log/train_epoch_loss_curve.jpg', bbox_inches='tight', dpi=450)

# show loss curve
plt.figure()
train_batch_loss = np.array(train_batch_loss)
train_batch_acc = np.array(train_batch_loss)
# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('epochs')  # x轴标签
plt.ylabel('loss')  # y轴标签
# 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
# 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
plt.plot(train_batch_loss[:, :1], train_batch_loss[:, 1:], linewidth=1, color='red', linestyle="solid",
         label="train batch loss")
plt.plot(train_batch_acc[:, :1], train_batch_acc[:, 1:], linewidth=1, color='blue', linestyle="solid",
         label="train batch loss")
plt.legend()
plt.title('Train Batches Loss Curve')
# plt.show()
plt.savefig('log/train_batch_loss_curve', bbox_inches='tight', dpi=450)
