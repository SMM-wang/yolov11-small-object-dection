import os


def count_class_labels(folder_path):
    # 初始化0到8的计数器
    class_counts = {i: 0 for i in range(9)}

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return

    # 获取文件夹内所有txt文件
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    print(f"在 '{folder_path}' 中找到 {len(txt_files)} 个txt文件。")

    for file_name in txt_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            # 获取每行的第一个数字（类别ID）
                            class_id = int(parts[0])

                            # 如果该数字在0-8范围内，进行统计
                            if 0 <= class_id <= 8:
                                class_counts[class_id] += 1
                        except ValueError:
                            # 如果转换失败（非数字），跳过该行
                            continue
        except Exception as e:
            print(f"读取文件 {file_name} 时出错: {e}")

    return class_counts


# 使用示例：请将路径替换为您实际的文件夹路径
# 例如: folder_path = r"C:\Users\Data\labels"
folder_path =r"C:\workspace\python\ultralytics-main\datasets\DsPCBSD+\valid\labels"  # 当前文件夹
results = count_class_labels(folder_path)

print("\n--- 统计结果 ---")
for class_id, count in results.items():
    print(f"数字 {class_id}: {count} 次")