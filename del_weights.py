import os
import shutil

folder_path1 = "./model_weights"  # 替换为实际的权重路径
folder_path2 = "./fig_results"  # 替换为实际的diff文件路径
# 遍历文件夹及其子文件夹中的所有文件

print("-------开始删除权重-------")
for root, dirs, files in os.walk(folder_path1):
    for file in files:
        file_path = os.path.join(root, file)  # 获取文件的完整路径
        if (file.endswith(".ckpt") and (file.startswith("unetformer") or file.startswith("last"))) or file.endswith(".txt"):
            os.remove(file_path)  # 删除文件
            print(f"已删除文件：{file_path}")
print("-------删除权重完成！-------")

print("-------开始删除diff文件-------")
# 遍历文件夹及其子文件夹中的所有文件和文件夹
for root, dirs, files in os.walk(folder_path2):
    for dir in dirs:
        if dir == "diff":
            dir_path = os.path.join(root, dir)  # 获取文件夹的完整路径
            shutil.rmtree(dir_path)  # 删除文件夹及其下的所有内容
            print(f"已删除文件夹：{dir_path}")
print("-------删除diff文件完成！-------")
print("-------程序结束！！！-------")
