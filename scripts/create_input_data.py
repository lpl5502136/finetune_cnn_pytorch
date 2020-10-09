#coding=utf-8
import os
import glob
import json
import sys


def data_dir_to_csv(indir, tag):
    f = open(tag+".csv", "w")
    cates = [x for x in os.listdir(indir) if not x.startswith(".")]  # mac 下有些隐藏文件以.开头
    print(cates)
    num_class = len(cates)

    # 建立类别映射
    cate_dict = {}
    for i, cate in enumerate(cates):
        cate_dict[cate] = i
    print(cate_dict)

    for cate in cates:
        sub_dir = os.path.join(indir, cate)
        img_list = glob.glob(os.path.join(sub_dir, "*.jpg"))
        cate_index = cate_dict[cate]
        for path in img_list:
            ww = path + "," + str(cate_index) + "\n"
            f.write(ww)
    f.close()


def check(train_dir, val_dir):
    """
    检查类别是否相等，并输出labels.json
    """
    train_cates = [x for x in os.listdir(train_dir) if not x.startswith(".")]
    val_cates = [x for x in os.listdir(val_dir) if not x.startswith(".")]
    if set(train_cates) == set(val_cates):
        d = {}
        for i, cate in enumerate(train_cates):
            d[cate] = i
        with open("labels.json", "w") as fp:
            fp.write(json.dumps(d))
        print(d)
    else:
        print("ERROR: train cates != val cates")


if __name__ == '__main__':
    """
    train 和 val 下面的子目录数量和类别必须一致 
    cd scripts
    python create_input_data.py /root/data/hymenoptera_data/train /root/data/hymenoptera_data/val
    """
    # train_dir = "/Users/lipeilun/Desktop/hymenoptera_data/train"
    # val_dir = "/Users/lipeilun/Desktop/hymenoptera_data/val"
    train_dir = sys.argv[1]
    val_dir = sys.argv[2]
    check(train_dir, val_dir)
    data_dir_to_csv(train_dir, "train")
    data_dir_to_csv(train_dir, "val")
