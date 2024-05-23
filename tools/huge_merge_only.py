import math
import os
import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
static_cols = -1
static_rows = -1


# 合并分割图像，指定行列数
# merge_path:切割后的小图片所在路径
# num_of_cols:多少列
# num_of_rows:多少行
# target_path:合成后的目标图片保存路径
# file_name:合成后的文件名称
def merge_only(merge_path, num_of_cols, num_of_rows, target_path, file_name):
    filename = os.listdir(merge_path)
    full_path = os.path.join(merge_path, filename[0])
    shape = cv2.imread(full_path).shape  # 三通道的影像需把-1改成1
    rows = shape[0]  # 高
    cols = shape[1]  # 宽
    channels = shape[2]

    dst = np.zeros((rows * num_of_rows, cols * num_of_cols, channels), np.uint8)
    for i in range(len(filename)):
        full_path = os.path.join(merge_path, filename[i])
        img = cv2.imread(full_path, 1)
        cols_th = int(full_path.split("_")[-1].split('.')[0])
        rows_th = int(full_path.split("_")[-2])
        roi = img[0:rows, 0:cols, :]
        dst[rows_th * rows:(rows_th + 1) * rows, cols_th * cols:(cols_th + 1) * cols, :] = roi
    cv2.imwrite(target_path + '/' + "merge-" + file_name, dst)


# 最终尺寸切割
# real_org_file_path:真正的原图所在的文件夹
# real_org_file_name:真正的原图的名称
# org_path:合成后的带有多余像素的图片所在路径
# fin_path:最终成品图保存的位置
# org_file_name:合成后带有多余像素的图片名字
# fin_file_name:最终成品图的名字
def cp(org_path, fin_path, org_file_name, fin_file_name):
    img = Image.open(org_path + '/' + org_file_name)
    real_img = Image.open(real_img_path)
    real_img = np.array(real_img)
    region = img.crop((0, 0, real_img.shape[1], real_img.shape[0]))  # 0,0表示要裁剪的位置的左上角坐标，高 h宽。
    region.save(fin_path + '/' + fin_file_name)  # 将裁剪下来的图片保存


if __name__ == '__main__':
    # merge_path:切割后的小图片所在路径
    # num_of_cols:多少列
    # num_of_rows:多少行
    # target_path:合成后的目标图片保存路径
    # file_name:合成后的文件名称
    # 01
    static_cols = 175
    static_rows = 183

    main_file_name = "huge_image_dpq.png"
    main_tmp_path = '/home/wym/projects/AAdatas/thz/predict/pre_baformer'
    real_img_path = "/home/wym/projects/AAdatas/dyz/label/dyz_label.tif"
    main_merge_path = '/home/wym/projects/BAFormer/data/thz/predict/tmp_superfluous'
    main_fin_path = '/home/wym/projects/BAFormer/data/thz/predict/final_output'

    # 合成
    print("running")
    merge_only(main_tmp_path, static_cols, static_rows, main_merge_path, main_file_name)
    # 裁剪
    cp(main_merge_path, main_fin_path, str("merge-" + main_file_name),main_file_name)
    print("done!!!")
