import torch.utils.data.distributed
import torchvision.transforms as transforms

from torch.autograd import Variable
import os
from PIL import Image
import json
from vit_pytorch.efficient import ViT
from linformer import Linformer

classes = ('GuideSign', 'M1', 'M4', 'M5', 'M6', 'M7', 'P1', 'P10_50', 'P12', 'W1')

transform_test = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

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
).to(DEVICE)

# model = torch.load("v1_vis_module.pth")
model.load_state_dict(torch.load("log/v5/v5_vis_modle_200.pth"))
# model.eval()  # 评估模式，而非训练模式
# model.to(DEVICE)
path = 'image_data/test_dataset_deblur/'  # 未去模糊的测试数据集
# path = 'image_data/test_dataset_deblur/' # 去模糊的测试数据集
testList = os.listdir(path)

# 保存结果为json格式
out_info = {}
out_info['annotations'] = []
for file in testList:
    img = Image.open(path + file)
    img = transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out = model(img)
    # Predict
    _, pred = torch.max(out.data, 1)
    print('Image Name:{},predict:{}'.format(file, classes[pred.data.item()]))
    out_data = {}
    out_data['filename'] = "test_dataset/" + file
    out_data['label'] = pred.data.item()
    out_info['annotations'].append(out_data)

with open("log/v5/deblur_test_dataset/22140615-侯志明-夏佳楠-PyTorch.json", 'w') as write_f:
    json.dump(out_info, write_f, indent=4, ensure_ascii=False)
