# def v():
#     num_ = [1,2]
#     lable_ = ['2','3']
#     acc_ = [3.0,4.0]
#     return num_, lable_, acc_
#
#
# num = []
# lable = []
# acc = []
#
# for i in range(10):
#     num_, lable_, acc_ = v()
#     num.extend(num_)
#     lable.extend(lable_)
#     acc.extend(acc_)
#
# print(num,lable,acc)
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# train_loss_points = []
# test_loss_points = []
#
#
# def data():
#     global train_loss_points
#     global test_loss_points
#     # [epoch, loss]
#     train_loss_points = [[1,1], [2,2], [3,3]]
#     test_loss_points = [[1,2], [2,3], [3,4]]
#
#
#
# def plot():
#     global train_loss_points
#     global test_loss_points
#     train_loss_points = np.array(train_loss_points)
#     test_loss_points = np.array(test_loss_points)
#     plt.figure()
#     plt.plot(train_loss_points[:, :1], train_loss_points[:, 1:], label="train_loss", linewidth=1.5)
#     plt.plot(test_loss_points[:, :1], test_loss_points[:, 1:], label="test_loss", linewidth=1.5)
#     plt.xlabel("epoch")
#     plt.ylabel("loss")
#     plt.legend()
#     plt.show()
#
# if __name__ == '__main__':
#     data()
#     plot()
#     plot()

print('sdfsf{}sdf'.format(2))