from model import *
from tensor import *
from utils import *
from my_optimizer import *
from mnist_test import *
import data_processing

test_img = './t10k-images.idx3-ubyte'
test_lab = './t10k-labels.idx1-ubyte'

test_images = data_processing.load_test_images(test_img)
test_labels = data_processing.load_test_labels(test_lab)


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


model_for_test = load("./mnist_test_model.pkl")  # 加载模型
T_full = TestModel(model_for_test,
                   test_img,
                   test_lab,
                   required_col=True,
                   test_size=100)  # 测试模型
print(f"Test accuracy: {T_full.show_accuracy()}")  # 打印准确率
print(f"Test number: {T_full.show_number()}")  # 显示测试的数字
print("confusion_matrix:")  # 显示混淆矩阵
T_full.show_confusion_matrix()
print(f"micro F1: {T_full.micro_F1()}")  # 显示micro F1
print(f"macro F1: {T_full.macro_F1()}")  # 显示macro F1
print("ROC curve:")  # 显示ROC曲线
T_full.roc_curve(100)
