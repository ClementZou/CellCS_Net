##### 环境依赖

python版本3.8.5

##### 文件结构

├── Readme.md 	//帮助文档

├──dataSet 	//数据集文件夹

├──log	 //训练日志文件夹

├──models 	//存放训练好的模型的文件夹

├──tensorBoard	 //tensorboard文件存储

├──matrixQ 	//用于保存代码中使用的Qinit

├──model.py	 //模型代码

├──test.py 	//测试脚本

├──train.py 	//训练脚本

├──utils.py	 //工具箱

└──requirements.txt 	//环境依赖

##### 启动方式

###### 训练集&测试集设置

按照如下架构将数据集保存至dataSet文件夹下

├── dataSet

│			├──数据集名称                      

│			│			├──A.mat

│			│			├──X.mat

│			│			└──Y.mat		

...		    ...

需要在train.py和test.py的指定位置编写用于选择部分数据进行训练或测试的代码，

###### 训练&测试

例如12次迭代，共训练200epoch，学习率为1e-5，在2号显卡上使用数据集“0429”进行训练，启用tensorBoard则使用如下命令进行训练

```
python train.py --layer_num 12 --gpu_list 2 --end_epoch 200 --learning_rate 1e-5 --data_dir 0429 --tensor_board True
```

例如12次迭代，使用数据集“0429”在学习率为1e-5下训练到第160epoch的结果，在数据集“0429”上使用2号显卡进行测试，则使用如下命令

```
--layer_num 12 --gpu_list 2 --epoch_num 160 --learning_rate 1e-4 --training_data_dir 0429 --test_data_dir 0429 
```

更多参数详见文件

###### 训练&测试结果

训练后的模型按照如下结构保存在models文件夹下

├──models

│			├──训练使用的数据集名称                      

│  		   │			├──模型名-数据类型（ref/sim）-学习率-通道数

│  		   │		     │		    ├──模型文件		

...			...			...			...

训练后的模型按照如下结构保存在result文件夹下

├──result

│			├──训练使用的数据集名称                      

│  		   │			├──模型名-数据类型（ref/sim）-学习率-通道数

│  		   │		     │		    ├──测试数据集名称		

│  		   │		     │		     │		    ├──n_grounTruth.png   //groundtruth图片

│  		   │		     │		     │		    ├──n_result.png			 //恢复结果

│  		   │		     │		     │		    ...

│  		   │		     │		     │		    └──log.txt 						//记录本次测试的相关信息

...			...			...			...			
