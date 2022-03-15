import operator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
def split_data_set(path, split_rate):
    list1 = pd.read_csv(path)
    list1 = shuffle(list1)
    total_length = len(list1)
    split_point = int(total_length * split_rate)
    list1 = list1.replace("Iris-setosa", "0")
    list1 = list1.replace("Iris-versicolor", "1")
    list1 = list1.replace("Iris-virginica", "2")
    x = list1.iloc[:, 0:4]
    x_train = x.iloc[:split_point, :]
    x_test = x.iloc[split_point:total_length + 1, :]
    y = list1.iloc[:, 4]
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:total_length + 1]
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
def data_diatance(x_test, x_train):
    distances = np.sqrt(sum((x_test - x_train) ** 2))
    return distances


def knn(x_test, x_train, y_train, k):
    predict_result_set = []
    train_set_size = len(x_train)
    distances = np.array(np.zeros(train_set_size))
    # 计算每一个测试集与每一个训练集的距离
    for i in x_test:
        for indx in range(train_set_size):
            # 计算数据之间的距离
            distances[indx] = data_diatance(i, x_train[indx])
        # 排序后的距离的下标
        sorted_dist = np.argsort(distances)
        class_count = {}
        # 取出k个最短距离
        for i in range(k):
            # 获得下标所对应的标签值
            sort_label = y_train[sorted_dist[i]]
            class_count[sort_label] = class_count.get(sort_label, 0) + 1
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        predict_result_set.append(sorted_class_count[0][0])

    return predict_result_set
def score(predict_result_set, y_test):
    count = 0
    for i in range(0, len(predict_result_set)):
        if predict_result_set[i] == y_test[i]:
            count += 1
    score = count / len(predict_result_set)
    return score
if __name__ == "__main__":
    # 载入数据集
    path = 'Iris.txt'
    split_rate = 0.3
    x_train, x_test, y_train, y_test=split_data_set(path,split_rate)
    X = []
    Y = []
    for k in range(2, 20):
        result = knn(x_test, x_train, y_train, k)
        # print("原有标签:", y_test)
        # # 为了方便对比查看，此处将预测结果转化为array,可直接打印结果
        # print("预测结果：", np.array(result))
        acc = score(result, y_test)
        X.append(k)
        Y.append(acc)
        # print("测试集的精度：%.2f" % acc)
    print(X, Y)
    plt.xlabel('k')
    plt.ylabel('acc')
    plt.plot(X, Y)
    plt.show()






