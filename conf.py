
##配置文件
conf = {

	#选择模型
	"model_name" : "cnn",

	#全局epoch
	"global_epochs" : 10,

	#本地epoch
	"local_epochs" : 10,


	"batch_size" : 64,

    #学习速率
	"lr" : 0.001,

	"momentum" : 0.9,

	#分类
	"num_classes": 10,

	#节点数
	"num_parties":10,

    #模型聚合权值
	"is_init_avg": True,

    #本地验证集划分比例
	"split_ratio": 0.3,

    #标签列名
	"label_column": "label",

	#数据列名
	"data_column": "file",

    #测试数据
	"test_dataset": "./data/cifar10/test/test.csv",

    #训练数据
	"train_dataset" : "./data/cifar10/train/train.csv",

    #模型保存目录
	"model_dir":"./save_model/",

    #模型文件名
	"model_file":"model.pth",

	"retrain":{
		"epoch": 10,
		"lr": 0.001,
		"num_vr":2000
	}
}