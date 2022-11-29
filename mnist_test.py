from model import *
import numpy as np
from tensor import *
import matplotlib.pyplot as plt
from utils import *
from my_optimizer import *
import data_processing
import random
from matplotlib.pyplot import MultipleLocator
import copy
import pandas as pd


class TestModel():
    def __init__(self,
                 model,
                 test_images_idx3_ubyte_file,
                 test_labels_idx1_ubyte_file,
                 required_col=False,
                 test_size=100):
        self.test_images = data_processing.load_test_images(
            test_images_idx3_ubyte_file)
        self.test_labels = data_processing.load_test_labels(
            test_labels_idx1_ubyte_file)

        self.img_test = []
        self.img_test_label = []
        self.test_size = test_size

        for i in range(test_size):
            if required_col:
                self.img_test.append(
                    data_processing.normalization(
                        data_processing.image2vector(self.test_images[i])))

            else:
                self.img_test.append(
                    data_processing.normalization(self.test_images[i])[None])
            label_test = np.zeros((10, 1))
            label_test[int(self.test_labels[i])] = 1
            self.img_test_label.append(label_test)

        print(self.img_test[0].shape, self.img_test_label[0].shape)

        self.test_model = model
        self.confusion_matrix = np.zeros((10, 10), dtype=int)
        self.correct_number = np.zeros(10, dtype=int)  # 每个数字的正确个数
        self.false_number = np.zeros(10, dtype=int)  # 每个数字的错误个数
        self.count_number = np.zeros(10, dtype=int)  # 每个数字的实际总个数
        self.data_list = []

        for m in range(test_size):
            my_input = tensor(self.img_test[m])
            out = self.test_model(my_input)
            self.confusion_matrix[np.argmax(out.data)][int(
                self.test_labels[m])] += 1
            self.count_number[int(self.test_labels[m])] += 1
            if np.argmax(out.data) == self.test_labels[m]:
                self.correct_number[int(self.test_labels[m])] += 1
            else:
                self.false_number[np.argmax(out.data)] += 1
            self.data_list.append(out.data)

    def accuracy(self):  # 各个数字的分别正确率
        return [
            float(self.correct_number[i] /
                  (self.correct_number[i] + self.false_number[i]))
            for i in range(10)
        ]

    def recall(self):  # 各个数字的分别召回值
        return [
            float(self.correct_number[i] / self.count_number[i])
            for i in range(10)
        ]

    def show_confusion_matrix(self):  # 显示混淆矩阵
        print(
            pd.DataFrame(self.confusion_matrix,
                         index=[f"Predict:{i}" for i in range(10)],
                         columns=[f"True:{i}" for i in range(10)]))

    def show_accuracy(self):  # 显示正确率
        c = self.accuracy()

        for i in range(10):
            print(f"number{i}_acc={c[i]}")

        plt.xlabel("number")
        plt.ylabel("accuracy")
        plt.plot(range(len(c)), c, color='green', linewidth=3)
        self.get_from_ten_number("ten number accuracy.png")

    def show_number(self):  # 显示各个数字的正确数量
        plt.xlabel("number")
        plt.ylabel("count")
        plt.plot(range(len(self.correct_number)),
                 self.correct_number,
                 color='red',
                 linewidth=3)
        plt.bar(range(len(self.count_number)), self.count_number, color='blue')
        self.get_from_ten_number("ten number correct number count.png")

    def get_from_ten_number(self, arg0):  # 从10个数字中获取信息
        x_major_locator = MultipleLocator(1)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.savefig(arg0)
        plt.show()

    def random_show(self, batch=6):  # 随机显示batch个图片
        ls = random.choices(self.test_images, k=batch)
        fig, ax = plt.subplots(
            nrows=1,
            ncols=batch,
            sharex=True,
            sharey=True,
        )
        ax = ax.flatten()
        for k, i in enumerate(ls):
            ax[k].imshow(i, cmap='Greys', interpolation='nearest')

            _input = tensor(data_processing.normalization(i)[None])
            out = self.test_model(_input)
            label = np.argmax(out.data)
            ax[k].set_title(f"{label}")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.show()

    def micro_accuracy(self):  # 微平均准确率
        return float(
            sum(self.correct_number) /
            (sum(self.correct_number) + sum(self.false_number)))

    def macro_accuracy(self):  # 宏平均准确率
        acc = self.accuracy()
        return sum(acc) / 10

    def micro_recall(self):  # 微平均召回率
        return float(sum(self.correct_number) / sum(self.count_number))

    def macro_recall(self):  # 宏平均召回率
        recall = self.recall()
        return sum(recall) / 10

    def weight_accuracy(self, weight_list):  # 加权平均准确率
        acc = self.accuracy()
        return sum(weight_list[i] * acc[i] for i in range(10))

    def weight_recall(self, weight_list):  # 加权平均召回率
        recall = self.recall()
        return sum(weight_list[i] * recall[i] for i in range(10))

    def micro_F1(self):  # 微平均F1
        return (2 * self.micro_accuracy() * self.micro_recall()) / (
            self.micro_recall() + self.micro_accuracy())

    def macro_F1(self):  # 宏平均F1
        return (2 * self.macro_accuracy() * self.macro_recall()) / (
            self.macro_recall() + self.macro_accuracy())

    def roc_curve(self, number):  # ROC曲线
        tpr = np.zeros((number, 10), dtype=float)
        fpr = np.zeros((number, 10), dtype=float)
        w = 0
        for x in range(number):
            tp_num = np.zeros(10, dtype=int)
            fp_num = np.zeros(10, dtype=int)
            for m in range(self.test_size):
                data = copy.deepcopy(self.data_list[m])
                num_1 = np.where(data >= w * data[np.argmax(data)])
                num_0 = np.where(data < w * data[np.argmax(data)])
                data[num_1] = 1
                data[num_0] = 0  
                for i in range(10):
                    if data[i] == 1:
                        if self.test_labels[m] == i:
                            tp_num[i] += 1
                        else:
                            fp_num[i] += 1
            for i in range(10):
                tpr[x][i] = float(tp_num[i] / self.count_number[i])
                fpr[x][i] = float(fp_num[i] /
                                  (self.test_size - self.count_number[i]))
            w += float(1 / number)


        color_list = [
            "#DC143C", "#8B008B", "#0000FF", "#2F4F4F", "#006400", "#FFFF00",
            "#DAA520", "#8B4513", "#FF0000", "#000000"
        ]
        for i in range(10):
            plt.plot(fpr[:, i],
                     tpr[:, i],
                     c=color_list[i],
                     lw=1,
                     alpha=0.75,
                     label=f'number:{i}')

        plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.8)  
        plt.xlim((-0.01, 1.02))
        plt.ylim((-0.01, 1.02))
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.grid(b=True, ls=':')
        plt.legend()
        plt.title(u'each_number ROC', fontsize=17)
        plt.show()