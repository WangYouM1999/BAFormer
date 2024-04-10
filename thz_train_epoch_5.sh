#!/bin/bash

# 定义要执行的次数
NUM_RUNS=5

# 循环执行命令
for ((i=1; i<=NUM_RUNS; i++))
do
    echo "执行第 $i 次训练,总计 $NUM_RUNS 次训练."
    python ./train_supervision.py -c ./config/thz/unetformer.py
    echo "第 $i 次训练完成,总计 $NUM_RUNS 次训练."
done