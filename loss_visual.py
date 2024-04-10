import matplotlib.pyplot as plt

# 从txt文件中读取数据
diff_path = "diff-0.4-85.44"
filename = "/home/wym/projects/unetformer-loss/fig_results/potsdam/" + diff_path + "/loss_epoch_log.txt"

# 初始化每列数据的列表和名称列表
columns = [[] for _ in range(4)]
column_names = ["main_loss", "aux_loss", "edge_loss", "all_loss"]

with open(filename, 'r') as file:
    for line in file:
        line = line.strip()
        if line:
            values = line.split(",")  # 如果数据以逗号分隔
            # values = line.split("\t")  # 如果数据以制表符分隔
            for i, val in enumerate(values):
                columns[i].append(float(val))

# 绘制折线图
for i in range(4):
    plt.plot(columns[i], label=column_names[i])

# 添加标题和标签
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')

# 添加图例
plt.legend()

# 显示并保存图形
save_dir = diff_path
save_path = save_dir + "/loss_value.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
