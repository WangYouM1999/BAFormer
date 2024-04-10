#!/bin/bash

folder_path1="./model_weights"  # 替换为实际的权重路径
folder_path2="./fig_results"  # 替换为实际的diff文件路径

echo "-------开始删除权重-------"
find "$folder_path1" -type f \( -name "*.ckpt" -a \( -name "adaptformer*" -o -name "last*" \) -o -name "*.txt" \) -print -delete
echo "-------删除权重完成！-------"

echo "-------开始删除diff文件-------"
find "$folder_path2" -type d \( -name "diff" \) -exec rm -rf {} + -print
echo "-------删除diff文件完成！-------"

echo "-------程序结束！！！-------"