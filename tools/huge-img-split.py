import math
import os
import cv2
import numpy as np
from PIL import Image
import torch
from osgeo import gdal

Image.MAX_IMAGE_PIXELS = None
static_cols = -1
static_rows = -1

# 清空TEMP
# dir_path:需要清空的目录路径
def del_files(dir_path):
    # os.walk会得到dir_path下各个后代文件夹和其中的文件的三元组列表，顺序自内而外排列，
    for root, dirs, files in os.walk(dir_path, topdown=False):
        # 第一步：删除文件
        for name in files:
            os.remove(os.path.join(root, name))  # 删除文件
        # 第二步：删除空文件夹
        for name in dirs:
            os.rmdir(os.path.join(root, name))  # 删除一个空目录
    print("clear done!!")


# 读取多光谱数据
def read_multispectral_image(image_path):
    dataset = gdal.Open(image_path)  # 打开.tif文件
    num_bands = dataset.RasterCount  # 获取波段数量
    proj = dataset.GetProjection()
    geotrans = dataset.GetGeoTransform()
    image_data = []
    for i in range(1, num_bands + 1):
        band = dataset.GetRasterBand(i)  # 获取波段
        band_data = band.ReadAsArray()  # 读取波段数据并转换为浮点型
        image_data.append(band_data)
    image_data = np.stack(image_data, axis=0)  # 将波段数据堆叠成一个多维数组
    image_data = image_data.transpose((1, 2, 0))  # 调整维度顺序为(H, W, C)
    # image_data /= 255.0  # 标准化像素值到0到1之间

    return image_data, proj, geotrans


def writeTiff(path, im_geotrans, im_proj, im_data):
    im_data = im_data.transpose((2, 0, 1))  # 调整维度顺序为(C, H, W)
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset
    print(f"保存影像到 {path} 成功！")


# 切割
# orgPath:加载图片所在路径
def qg(orgPath):
    del_files(" ")
    global static_cols
    global static_rows

    # 目标分割大小
    DES_HEIGHT = 512
    DES_WIDTH = 512

    # 获取图像信息
    path_img = orgPath

    # 获取原始高分辨的图像的属性信息
    # 图像
    # src = cv2.imread(path_img, 1)
    img, proj, geotrans = read_multispectral_image(path_img)
    # src = Image.open(path_img)
    height = img.shape[0]
    width = img.shape[1]

    # 把原始图像边缘填充至分割大小的整数倍
    pad_height = math.ceil(height / DES_HEIGHT) * DES_HEIGHT - height
    pad_width = math.ceil(width / DES_WIDTH) * DES_WIDTH - width
    # 使用np.pad函数进行填充
    img = np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
    sum_rows = img.shape[0]  # 高度
    sum_cols = img.shape[1]  # 宽度

    cols = DES_WIDTH
    rows = DES_HEIGHT
    static_rows = int(sum_rows / rows)
    static_cols = int(sum_cols / cols)
    filename = os.path.split(path_img)[1]
    for i in range(int(sum_cols / cols)):
        for j in range(int(sum_rows / rows)):
            # cv2.imwrite(main_tmp_path + "/" + os.path.splitext(filename)[0] + '_' + str(j) + '_' + str(i) + '.png',
            #             img[j * rows:(j + 1) * rows, i * cols:(i + 1) * cols, :], )
            writeTiff(main_tmp_path + "/" + os.path.splitext(filename)[0] + '_' + str(j) + '_' + str(i) + '.tif',
                      geotrans, proj, img[j * rows:(j + 1) * rows, i * cols:(i + 1) * cols, :])
    return height, width

# 合并分割图像，指定行列数
# merge_path:切割后的小图片所在路径
# num_of_cols:多少列
# num_of_rows:多少行
# target_path:合成后的目标图片保存路径
# file_name:合成后的文件名称
def merge_picture(merge_path, num_of_cols, num_of_rows, target_path, file_name):
    filename = os.listdir(merge_path)
    full_path = os.path.join(merge_path, filename[0])
    img, proj, geotrans = read_multispectral_image(full_path)
    cols = img.shape[1]
    rows = img.shape[0]
    channels = img.shape[2]

    dst = np.zeros((rows * num_of_rows, cols * num_of_cols, channels))
    for i in range(len(filename)):
        full_path = os.path.join(merge_path, filename[i])
        img = cv2.imread(full_path, 1)
        cols_th = int(full_path.split("_")[-1].split('.')[0])
        rows_th = int(full_path.split("_")[-2])
        roi = img[0:rows, 0:cols, :]
        dst[rows_th * rows:(rows_th + 1) * rows, cols_th * cols:(cols_th + 1) * cols, :] = roi
    writeTiff(target_path + "/" + "merge-" + file_name, geotrans, proj, dst)


# 最终尺寸切割
# org_path:合成后的带有多余像素的图片所在路径
# fin_path:最终成品图保存的位置
# org_file_name:合成后带有多余像素的图片名字
# fin_file_name:最终成品图的名字
def cp(file_path, final_path, file_name, final_file_name, real_height, real_width):
    img, proj, geotrans = read_multispectral_image(file_path + '/' + file_name)
    # region = img.crop((0, 0, Real_Height, Real_Height))  # 0,0表示要裁剪的位置的左上角坐标，高 h宽。
    img = img[0:real_height, 0:real_width, :]
    writeTiff(final_path + '/' + final_file_name, geotrans, proj, img)  # 将裁剪下来的图片保存
    # img = Image.open(file_path + "/" + file_name)  # 打开chess.png文件，并赋值给img
    # region = img.crop((0, 0, Real_Height, Real_Height))  # 0,0表示要裁剪的位置的左上角坐标，w长 h宽。
    # region.save(final_path + "/" + final_file_name)  # 将裁剪下来的图片保存


# 加载图片列表
def round_read_file(file_path):
    image_name_list = []
    for file_name in os.listdir(file_path):
        # 排除MacOS的.DS_Store文件，Windows下不受影响。
        if file_name != '.DS_Store':
            print("加载图片:" + str(file_path) + '/' + str(file_name))
            image_name_list.append(str(file_name))
    print("总计加载图片数量" + str(len(image_name_list)))
    return image_name_list


# 主函数
if __name__ == '__main__':
    main_tmp_path = '/home/wym/projects/AAdatas/thz/img_split/tmp_output'  # 切割后的照片的存储路径
    main_org_path = '/home/wym/projects/AAdatas/thz/img_split/test_img'  # 文件目录
    main_merge_path = '/home/wym/projects/AAdatas/thz/img_split/tmp_superfluous'  # 带边合成图像
    main_fin_path = '/home/wym/projects/AAdatas/thz/img_split/final_output'  # 裁剪后最终合成图像
    real_img_path = "/home/wym/projects/AAdatas/thz/img_split/test_img/thz.tif"
    main_img_name_list = round_read_file(main_org_path)

    i = 1
    for main_file_name in main_img_name_list:
        # 转成什么格式
        target_name = main_file_name.split('.')[0] + '.tif'
        print("第" + str(i) + "/" + str(len(main_img_name_list)) + "项，" + "当前图片名称：" + main_file_name)
        real_height, real_width = qg(main_org_path + "/" + main_file_name)
        print("第" + str(i) + "/" + str(len(main_img_name_list)) + "项，切割完成，当前row：" + str(
            static_rows) + ",col：" + str(static_cols))
        ################################################
        # 可在此处添加对小图的处理，例如某些对图片尺寸有要求的API#
        ################################################
        merge_picture(main_tmp_path, static_cols, static_rows, main_merge_path, target_name)
        print("第" + str(i) + "/" + str(len(main_img_name_list)) + "项，合并完成")
        cp(main_merge_path, main_fin_path, str("merge-" + target_name), target_name, real_height, real_width)
        print("第" + str(i) + "/" + str(len(main_img_name_list)) + "项，完成")
        static_rows = -1
        static_cols = -1
        i = i + 1
    print("done!!!!")
