import numpy as np
import pandas as pd
from conf import conf
from fedavg.datasets import MyTabularDataset
from sklearn.utils import shuffle

def label_skew(data,label,K,n_parties,beta,min_require_size = 10):
    """
    :param data: 数据dataframe
    :param label: 标签列名
    :param K: 标签数
    :param n_parties:参与方数
    :param beta: 狄利克雷参数
    :param min_require_size: 点最小数据量，如果低于这个数字会重新划分，保证每个节点数据量不会过少
    :return: 根据狄利克雷分布划分数据到各个参与方
    """
    y_train = data[label]

    min_size = 0

    N = y_train.shape[0]  # N样本总数
    # return train_datasets, test_dataset, n_input, number_samples
    split_data = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])

            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

            min_size = min([len(idx_j) for idx_j in idx_batch])

    # 根据各节点数据index划分数据
    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        split_data[j] = data.iloc[idx_batch[j], :]

    return split_data


def get_tabular_data():
    """
    :return: 加载数据
    """
    ###训练数据路径
    train_dataset_file = conf["train_dataset"]
    #测试数据路径
    test_dataset_file = conf["test_dataset"]

    train_datasets = {}
    val_datasets = {}
    ##各节点数据量
    number_samples = {}

    ##读取数据集,训练数据拆分成训练集和测试集
    for key in train_dataset_file.keys():
        train_dataset = pd.read_csv(train_dataset_file[key])

        val_dataset = train_dataset[:int(len(train_dataset)*conf["split_ratio"])]
        train_dataset = train_dataset[int(len(train_dataset)*conf["split_ratio"]):]
        train_datasets[key] = MyTabularDataset(train_dataset, conf["label_column"])
        val_datasets[key] = MyTabularDataset(val_dataset,conf["label_column"])

        number_samples[key] = len(train_dataset)

    ##测试集,在Server端测试模型效果
    test_dataset = pd.read_csv(test_dataset_file)

    #模型输入维度
    n_input = test_dataset.shape[1] - 1
    test_dataset = MyTabularDataset(test_dataset,conf["label_column"])
    print("数据加载完成!")

    return train_datasets, val_datasets, test_dataset, n_input


def get_cifar10():

    ###训练数据
    train_data = pd.read_csv(conf["train_dataset"])

    train_data = label_skew(train_data,conf["label_column"],conf["num_classes"],conf["num_parties"],0.1)

    train_datasets = {}
    val_datasets = {}
    ##各节点数据量
    number_samples = {}

    ##读取数据集,训练数据拆分成训练集和测试集
    for key in train_data.keys():
        ##打乱顺序
        train_dataset = shuffle(train_data[key])

        val_dataset = train_dataset[:int(len(train_dataset) * conf["split_ratio"])]
        train_dataset = train_dataset[int(len(train_dataset) * conf["split_ratio"]):]
        train_datasets[key] = train_dataset
        val_datasets[key] = val_dataset

        number_samples[key] = len(train_dataset)

    ##测试集,在Server端测试模型效果
    test_dataset = pd.read_csv(conf["test_dataset"])
    test_dataset = test_dataset
    print("数据加载完成!")

    return train_datasets, val_datasets, test_dataset




if __name__ == "__main__":
    # trainset = pd.read_csv('./data/cifar10/train/train.csv')
    # label_skew(trainset,'label',10,3,0.1,'./data/cifar10/train/')



    a = [1]
    b = a*256
    print(b)



