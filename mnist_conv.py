import matplotlib.pyplot as plt
import data_processing
from model import *
from typing import OrderedDict
import numpy as np
from tensor import *
import matplotlib.pyplot as plt
from utils import *
from my_optimizer import *
import utils

train_images_idx3_ubyte_file = './train-images.idx3-ubyte'
train_labels_idx1_ubyte_file = './train-labels.idx1-ubyte'
test_images_idx3_ubyte_file = './t10k-images.idx3-ubyte'
test_labels_idx1_ubyte_file = './t10k-labels.idx1-ubyte'
#加载
train_images = data_processing.load_train_images(train_images_idx3_ubyte_file)
train_labels = data_processing.load_train_labels(train_labels_idx1_ubyte_file)
test_images = data_processing.load_test_images(test_images_idx3_ubyte_file)
test_labels = data_processing.load_test_labels(test_labels_idx1_ubyte_file)

train_size = 200
img_train, img_label = data_processing.pre_process(train_images, train_labels,
                                                   train_size)

print(img_train[0].shape, img_label[0].shape)


# The class A is a subclass of the Module class. It has a constructor that initializes the superclass
# and then initializes the parameters of the class. The forward function is the function that is
# called when the class is called. It takes an input and returns an output
class A(Module):
    def __init__(self):
        super(A, self).__init__()
        self.fc1 = Conv2d(1, 16, (5, 5))
        self.f1 = Max_pooling(2)
        self.fc2 = Conv2d(16, 32, (5, 5))
        self.f2 = Max_pooling(2) 
        self.fc3 = Dims_change()
        self.fun1 = Sigmoid()
        self.fc4 = Linear(512, 10, RANDOM)
        self.f4 = Softmax()

    def forward(self, input):
        x = self.fc1(input)
        x = self.f1(x)

        x = self.fc2(x)
        x = self.f2(x)

        x = self.fc3(x)

        x = self.fun1(x)
        x = self.fc4(x)
        x = self.f4(x)
        return x


# sourcery skip: avoid-builtin-shadow
epoh = 10
r = 0.01
model = A()
opti = Optim()
opti = SGD(model, r)
loss = cross_entropy()
draw_loss = utils.draw_function(epoh)  #动态绘图装饰器

for i in range(epoh):
    for j in range(train_size):
        input = tensor(img_train[j])
        out = model(input)
        opti.Loss(out, tensor(img_label[j]), loss)
        opti.Step()
        opti.zero_grad()

    c = 0
    for m in range(train_size):
        input = tensor(img_train[m])
        out = model(input)
        if np.argmax(out.data) == train_labels[m]:
            c += 1
    #绘图
    draw_loss(opti.loss, i == epoh)
    print(f"Epoh:{i}  Acc:{c / train_size}  Loss:{opti.loss}")

save(model, './model_conv.pkl')
