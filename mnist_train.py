from model import *
import numpy as np
from tensor import *
from my_optimizer import *
import random
import data_processing
from utils import *

train_images_idx3_ubyte_file = './train-images.idx3-ubyte'
train_labels_idx1_ubyte_file = './train-labels.idx1-ubyte'
test_images_idx3_ubyte_file = './t10k-images.idx3-ubyte'
test_labels_idx1_ubyte_file = './t10k-labels.idx1-ubyte'


train_images = data_processing.load_train_images(train_images_idx3_ubyte_file)
train_labels = data_processing.load_train_labels(train_labels_idx1_ubyte_file)
test_images = data_processing.load_test_images(test_images_idx3_ubyte_file)
test_labels = data_processing.load_test_labels(test_labels_idx1_ubyte_file)

img_train = []
img_label = []
train_size = 1000
max_epochs = 200
batch_size = 10

for i in range(train_size):
    img_train.append(
        data_processing.normalization(data_processing.image2vector(train_images[i])))
    label_train = np.zeros((10, 1))
    label_train[int(train_labels[i])] = 1
    img_label.append(label_train)



class MnistModel(Module):
    def __init__(self, input, hid_1, hid_2, hid_3, output):
        super(MnistModel, self).__init__()
        self.fc1 = Linear(input, hid_1, RANDOM)
        self.f1 = Sigmoid()
        self.fc2 = Linear(hid_1, hid_2, RANDOM)
        self.f2 = Sigmoid()
        self.fc3 = Linear(hid_2, hid_3, RANDOM)
        self.f3 = Sigmoid()
        self.fc4 = Linear(hid_3, output, RANDOM)
        self.f4 = Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.f1(x)
        x = self.fc2(x)
        x = self.f2(x)
        x = self.fc3(x)
        x = self.f3(x)
        x = self.fc4(x)
        x = self.f4(x)
        return x


model = MnistModel(784, 128, 64, 32, 10) # 输入参数
optimizer = Optim() # 优化器
optimizer = SGD(model, 0.1, minibatch=batch_size) # 优化器
loss_function = cross_entropy() # 损失函数
draw_loss = draw_function() # 画图函数
ls = list(range(train_size)) # 用于打乱数据集

for i in range(max_epochs):
    random.shuffle(ls)
    for j in range(0, train_size, batch_size):
        _input, label = data_processing.rand_select(ls[j:j + batch_size], img_train,
                                           img_label)
        _output = model(tensor(_input))
        optimizer.Loss(_output, tensor(label), loss_function) # 计算损失
        optimizer.Step() # 更新参数
        optimizer.zero_grad() # 梯度清零
    draw_loss(optimizer.loss, i == max_epochs)
    print(f"epoch:{i}  loss:{optimizer.loss}") # 打印损失

save(model, './mnist_test_model.pkl') # 保存模型

