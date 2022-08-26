import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

classes = ('GuideSign', 'M1', 'M4', 'M5', 'M6', 'M7', 'P1', 'P10_50', 'P12', 'W1')
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
model = torch.load("v1_vis_module.pth")
# model.eval()
# model.to(DEVICE)

dataset_test = datasets.ImageFolder('test_dataset', transform_test)
print(len(dataset_test))
# 对应文件夹的label

for index in range(len(dataset_test)):
    item = dataset_test[index]
    img, label = item
    img.unsqueeze_(0)
    data = Variable(img).to(DEVICE)
    output = model(data)
    _, pred = torch.max(output.data, 1)
    print('Image Name:{},predict:{}'.format(dataset_test.imgs[index][0], classes[pred.data.item()]))
