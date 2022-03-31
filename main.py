import json,os
import pandas as pd
from conf import conf
import torch
import numpy as np
from fedavg.server import Server
from fedavg.client import Client

from fedavg.models import CNN_Model,weights_init_normal, ReTrainModel
from utils import get_cifar10
import copy


if __name__ == '__main__':

    train_datasets, val_datasets, test_dataset = get_cifar10()

    ###初始化每个节点聚合权值
    client_weight = {}
    if conf["is_init_avg"]:
        for key in train_datasets.keys():
            client_weight[key] = 1 / len(train_datasets)

    print("聚合权值初始化")

    ##保存节点
    clients = {}
    # 保存节点模型
    clients_models = {}

    ##训练目标模型
    model = CNN_Model()
    model.apply(weights_init_normal)
    if torch.cuda.is_available():
        model.cuda()

    server = Server(conf, model, test_dataset)

    print("Server初始化完成!")

    for key in train_datasets.keys():
        clients[key] = Client(conf, server.global_model, train_datasets[key], val_datasets[key])

    print("参与方初始化完成！")

    # 保存模型
    if not os.path.isdir(conf["model_dir"]):
        os.mkdir(conf["model_dir"])
    max_acc = 0

    #联邦训练
    for e in range(conf["global_epochs"]):

        for key in clients.keys():
            print('training client {}...'.format(key))
            model_k = clients[key].local_train(server.global_model)
            clients_models[key] = copy.deepcopy(model_k)

        #联邦聚合
        server.model_aggregate(clients_models, client_weight)
        #测试全局模型
        acc, loss = server.model_eval()
        print("Epoch %d, global_acc: %f, global_loss: %f\n" % (e, acc, loss))

        #保存最好的模型
        if acc >= max_acc:
            torch.save(server.global_model.state_dict(), os.path.join(conf["model_dir"], "model-epoch{}.pth".format(e)))
            max_acc = acc

    #使用VR进行后处理
    client_mean = {}
    client_cov = {}
    client_length = {}

    for key in clients.keys():
        #计算局部特征均值和协方差
        c_mean, c_cov, c_length = clients[key].cal_distributions(server.global_model)
        client_mean[key] = c_mean
        client_cov[key] = c_cov
        client_length[key] = c_length
    print("完成局部特征均值和协方差计算")


    #计算全局均值和协方差
    g_mean, g_cov = server.cal_global_gd(client_mean, client_cov, client_length)
    print("完成全局均值和协方差计算")

    #生成虚拟特征
    print("")
    retrain_vr = []
    label = []
    eval_vr = []
    for i in range(conf['num_classes']):
        mean = np.squeeze(np.array(g_mean[i]))
        vr = np.random.multivariate_normal(mean, g_cov[i], conf["retrain"]["num_vr"]*2)
        retrain_vr.extend(vr.tolist()[:conf["retrain"]["num_vr"]])
        eval_vr.extend(vr.tolist()[conf["retrain"]["num_vr"]:])
        label.extend([i]*conf["retrain"]["num_vr"])

    print("完成虚拟特征生成")

    #获取要重来的网络层
    retrain_model = ReTrainModel()
    if torch.cuda.is_available():
        retrain_model.cuda()
    reset_name = []
    for name, _ in retrain_model.state_dict().items():
        reset_name.append(name)

    #初始化重训练模型
    for name, param in server.global_model.state_dict().items():
        if name in reset_name:
            retrain_model.state_dict()[name].copy_(param.clone())

    #使用VR进行重训练
    retrain_model = server.retrain_vr(retrain_vr, label, eval_vr, retrain_model)
    print("完成重训练")

    ## 使用重训练网络层 更新 全局模型
    for name, param in retrain_model.state_dict().items():
        server.global_model.state_dict()[name].copy_(param.clone())

    acc, loss = server.model_eval()
    print("After retraining global_acc: %f, global_loss: %f\n" % (acc, loss))


    torch.save(server.global_model.state_dict(), os.path.join(conf["model_dir"],conf["model_file"]))

    print("联邦训练完成，模型保存在{0}目录下!".format(conf["model_dir"]))