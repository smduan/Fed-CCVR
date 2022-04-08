
##配置文件
conf = {

	#数据类型，tabular, image
	"data_type" : "tabular",

	#选择模型mlp,simple-cnn,vgg
	"model_name" : "mlp",

	#处理方法:fed_ccvr
	"no-iid": "",

	#全局epoch
	"global_epochs" : 1000,

	#本地epoch
	"local_epochs" : 3,

	#狄利克雷参数
	"beta" : 0.5,

	"batch_size" : 64,

	"weight_decay":1e-5,

    #学习速率
	"lr" : 0.001,

	"momentum" : 0.9,

	#分类
	"num_classes": 2,

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
	"test_dataset": "./data/adult/adult_test.csv",

    #训练数据
	"train_dataset" : "./data/adult/adult_train.csv",

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