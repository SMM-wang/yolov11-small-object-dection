import os
import shutil
import random
from tqdm import tqdm  # 用于显示进度条 (pip install tqdm)

# ================= 配置区域 =================
# 原始数据集路径
source_images_root = r"C:\workspace\python\ultralytics-main\datasets\PCB_DATASET\images"  # 图片源根目录 (内含6个分类文件夹)
source_labels_root = r"C:\workspace\python\ultralytics-main\datasets\PCB_DATASET\Annotations"  # 标注源根目录 (内含6个分类文件夹)

# 目标数据集根目录 (脚本会自动创建)
target_root = r"C:\workspace\python\ultralytics-main\datasets\PCB_DATASET"

# 划分比例 (训练集 : 验证集 : 测试集)
split_ratio = [0.8, 0.15, 0.05]

# 支持的图片扩展名 (防止读取到隐藏文件等)
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']


# ===========================================

def make_dirs():
    """创建目标文件夹结构"""
    sub_sets = ['train', 'val', 'test']
    sub_dirs = ['images', 'labels']

    for s in sub_sets:
        for d in sub_dirs:
            path = os.path.join(target_root, s, d)
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"Created folder: {path}")


def split_dataset():
    # 1. 检查源文件夹是否存在
    if not os.path.exists(source_images_root) or not os.path.exists(source_labels_root):
        print("错误：找不到源文件夹，请检查路径配置。")
        return

    make_dirs()

    # 2. 获取所有类别 (子文件夹名称)
    classes = [d for d in os.listdir(source_images_root) if os.path.isdir(os.path.join(source_images_root, d))]
    print(f"检测到 {len(classes)} 个类别: {classes}")

    total_images_processed = 0

    # 3. 遍历每个类别进行分层划分
    for cls in classes:
        cls_img_dir = os.path.join(source_images_root, cls)
        cls_xml_dir = os.path.join(source_labels_root, cls)

        if not os.path.exists(cls_xml_dir):
            print(f"警告：类别 {cls} 在 Annotations 中没有对应的文件夹，跳过。")
            continue

        # 获取该类别下所有图片文件
        images = [f for f in os.listdir(cls_img_dir) if os.path.splitext(f)[-1].lower() in valid_extensions]

        # 验证对应的XML是否存在，确保一一对应
        valid_pairs = []
        for img_name in images:
            file_id = os.path.splitext(img_name)[0]
            xml_name = file_id + ".xml"
            xml_path = os.path.join(cls_xml_dir, xml_name)

            if os.path.exists(xml_path):
                valid_pairs.append((img_name, xml_name))
            else:
                print(f"警告: 图片 {img_name} 缺少对应的 XML 文件，已跳过。")

        # 随机打乱
        random.shuffle(valid_pairs)

        num_total = len(valid_pairs)
        num_train = int(num_total * split_ratio[0])
        num_val = int(num_total * split_ratio[1])
        # 测试集取剩余部分，确保总和一致
        num_test = num_total - num_train - num_val

        print(f"正在处理类别 [{cls}]: 总数 {num_total} -> Train: {num_train}, Val: {num_val}, Test: {num_test}")

        # 执行复制操作
        # 定义数据集划分的边界
        train_pairs = valid_pairs[:num_train]
        val_pairs = valid_pairs[num_train: num_train + num_val]
        test_pairs = valid_pairs[num_train + num_val:]

        sets = [
            ('train', train_pairs),
            ('val', val_pairs),
            ('test', test_pairs)
        ]

        for set_name, pairs in sets:
            # 使用 tqdm 显示进度条，如果不需要可直接用 for pair in pairs:
            for img_file, xml_file in tqdm(pairs, desc=f"Moving {cls} to {set_name}", leave=False):
                # 源路径
                src_img_path = os.path.join(cls_img_dir, img_file)
                src_xml_path = os.path.join(cls_xml_dir, xml_file)

                # 目标路径 (不再包含 cls 子文件夹，直接扁平化)
                dst_img_path = os.path.join(target_root, set_name, 'images', img_file)
                dst_xml_path = os.path.join(target_root, set_name, 'labels', xml_file)

                # 复制文件
                shutil.copy2(src_img_path, dst_img_path)
                shutil.copy2(src_xml_path, dst_xml_path)

        total_images_processed += num_total

    print("-" * 30)
    print(f"处理完成！共处理 {total_images_processed} 组数据。")
    print(f"数据集已保存在: {os.path.abspath(target_root)}")


if __name__ == "__main__":
    split_dataset()