#!/bin/bash

# 捕获终止信号并终止所有子进程
trap 'kill $(jobs -p)' EXIT

n_cite=0
echo "n_cite: $n_cite"

num_chunks=1

echo "num_chunks: $num_chunks"
# echo "每个任务处理的数据量: $chunks_size"

# 创建结果目录
mkdir -p "./res/paper_c$((n_cite+1))"

# 使用 ProcessPoolExecutor 并行运行多个 Python 进程
python_script="quantize_v3.py"

st=0
ed=0

echo "start from $st to $ed"

for i in $(seq $st $ed); do
    python $python_script --n_chunk $i --n_cite $n_cite --total_chunk $num_chunks &
    sleep 5
done

# 等待所有后台进程完成
wait

echo "所有任务已完成"
