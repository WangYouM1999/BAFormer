#!/bin/bash

# 定义起始和结束的权重参数
start_index=1
end_index=1

# 创建权重参数列表
WEIGHTS=("adaptformer-init-r18-512crop-ms-epoch30-rep")  # 将起始权重参数添加到列表中
for ((i=start_index; i<end_index; i++))
do
#    if ((i == 0)); then
#        weight="${WEIGHTS[0]}"
#    else
#        weight="${WEIGHTS[0]}-v$i"
#    fi
    WEIGHTS+=("$weight")
done
# 在数组最后加入名为"last"的权重
#WEIGHTS+=("last")

# 计算权重总数
total=${#WEIGHTS[@]}

# 获取文件名中的前缀
file_prefix=$(echo "$0" | awk -F '_' '{print $1}')

# 创建保存测试精度信息的文件
result_file="model_weights/loveda/adaptformer-init-r18-512crop-ms-epoch30-rep/${file_prefix}_test_results.txt"
> $result_file

echo "开始执行测试，共需测试 $total 个权重参数"

# 循环执行测试命令
index=0
progress_bar=""

miou_array=()  # 存储提取的 mIOU 值

for weight in "${WEIGHTS[@]}"
do
    index=$(( index + 1 ))

    # 更新权重参数
    sed -i "s/test_weights_name = \".*\"/test_weights_name = \"$weight\"/g" ./config/loveda/adaptformer.py

    # 执行测试命令，并将精度信息追加写入文件
    echo "执行测试，权重参数为: $weight.ckpt"
    echo "------------------------------------------------------"
    result=$(python ./loveda_test.py -c ./config/loveda/adaptformer.py -o fig_results/loveda/diff/adaptformer -t 'd4')
    echo "权重参数为: $weight.ckpt" >> $result_file
    echo "$result" >> $result_file
    echo "" >> $result_file
    echo "------------------------------------------------------"

    # 提取并保存 mIOU 值
    extracted_miou=$(echo "$result" | awk -F ', mIOU:' '{print $2}' | awk '{print $1}')
    miou_array+=("$extracted_miou")

    # 构建进度条字符串
    progress_bar="["
    for ((j=0; j<index; j++))
    do
        progress_bar+="#"
    done
    for ((j=index; j<total; j++))
    do
        progress_bar+=" "
    done
    progress_bar+="]"

    # 显示进度条
    progress=$(($index * 100 / $total))
    printf "[%3d%%] 进度: %s \r" "$progress" "$progress_bar"
done

echo # 打印换行符

# 将提取的 mIOU 值按照每三个值一行的格式保存到文件
echo "------------------------------------------------------"
count=1
line_num=0
echo -n "第$((line_num+1))次训练mIOU：" >> $result_file
for miou in "${miou_array[@]}"
do
    miou_formatted=$(printf "%.4f" $miou)
    echo -n " $miou_formatted" >> $result_file
    if [ $count -eq 3 ]; then
        line_num=$((line_num+1))
        echo "，" >> $result_file
        echo -n "第$((line_num+1))次训练mIOU：" >> $result_file
        count=1
    else
        echo -n "、" >> $result_file
        count=$((count+1))
    fi
done

# 如果还有剩余的不足三个 mIOU 值，则追加到文件末尾
if [ $count -ne 1 ]; then
    echo "" >> $result_file
fi

# 显示测试精度信息
echo "测试精度信息已保存到文件: $result_file"
echo "------------------------------------------------------"
cat $result_file
echo "------------------------------------------------------"