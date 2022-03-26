import torch
import numpy as np
from fedavg.datasets import MyImageDataset

class Client(object):

    def __init__(self, conf, model, train_df, val_df):
        """
        :param conf: 配置文件
        :param model: 全局模型
        :param train_dataset: 训练数据集
        :param val_dataset: 验证数据集
        """

        self.conf = conf

        self.local_model = model
        self.train_df = train_df
        self.train_dataset = MyImageDataset(self.train_df,conf["data_column"], conf["label_column"])
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],shuffle=True)

        self.val_df = val_df
        self.val_dataset = MyImageDataset(self.val_df,conf["data_column"], conf["label_column"])
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=conf["batch_size"],shuffle=True)

    def local_train(self, model):

        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])
        # optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.conf['lr'])
        criterion = torch.nn.CrossEntropyLoss()
        for e in range(self.conf["local_epochs"]):
            self.local_model.train()
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                feature, output = self.local_model(data)

                loss = criterion(output, target)
                loss.backward()

                optimizer.step()

            acc, eval_loss = self.model_eval()
            print("Epoch {0} done. train_loss ={1}, eval_loss = {2}, eval_acc={3}".format(e, loss, eval_loss, acc))

        return self.local_model.state_dict()

    def model_eval(self):
        self.local_model.eval()

        total_loss = 0.0
        correct = 0
        dataset_size = 0
        predict = []
        label = []
        criterion = torch.nn.CrossEntropyLoss()
        for batch_id, batch in enumerate(self.val_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            _, output = self.local_model(data)

            total_loss += criterion(output, target)    # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability

            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

            predict.extend(pred.numpy())
            label.extend(target.numpy())
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc, total_l

    def _cal_mean_cov(self,features):
        """
        :param features: 输出特征，(batch_size, 256)
        :return:
        """
        features = np.array(features)
        mean = np.mean(features, axis=0)
        cov = np.cov(features.T)
        return mean,cov

    def cal_distributions(self, model):
        """
        :param feature:
        :return:
        """
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        self.local_model.eval()

        features = []
        mean = []
        cov = []
        length = []

        for i in range(self.conf["num_classes"]):
            train_i = self.train_df[self.train_df[self.conf['label_column']] == i]
            train_i_dataset = MyImageDataset(train_i, self.conf['data_column'], self.conf['label_column'])

            if len(train_i_dataset) > 0:
                train_i_loader = torch.utils.data.DataLoader(train_i_dataset, batch_size=self.conf["batch_size"],
                                                             shuffle=True)
                for batch_id, batch in enumerate(train_i_loader):
                    data, target = batch

                    if torch.cuda.is_available():
                        data = data.cuda()

                    feature, _ = self.local_model(data)
                    features.extend(feature.tolist())

                f_mean, f_cov = self._cal_mean_cov(features)

            else:
                ##TODO: 根据最后一层隐藏层输出决定均值和方差的维度
                f_mean = np.zeros((256,))
                f_cov = np.zeros((256,256))

            mean.append(f_mean)
            cov.append(f_cov)
            length.append(len(train_i))

        return mean, cov, length







