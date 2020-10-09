# 微调一个CNN网络
微调CNN网络的训练代码
前向传播代码
# 准备
下载数据集
转化成需要的格式:
python create_input_data.py /root/data/hymenoptera_data/train /root/data/hymenoptera_data/val
scripts/labels.json
scripts/train.csv
scripts/val.csv
# 训练
python train.py
# 推理
python inference.py ./output/best_model.pth
# finetune_cnn_pytorch
