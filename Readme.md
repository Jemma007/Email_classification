question9.py为模型主要文件

preprocess用于生成data中的三个csv文件

email-classification.ipynb为在kaggle上使用gpu运行结果

将压缩包解压到data文件夹中即可使用preprocess.py文件生成train.csv, test.csv, valid.csv文件

运行时注意使用torchtext==0.9.0
