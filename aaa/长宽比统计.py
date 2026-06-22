import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def analyze_yolo_aspect_ratios(label_dir, percentile_clip=99):
    """
    分析 YOLO 格式数据集的长宽比 (w/h)
    
    参数:
        label_dir (str): YOLO 标签(.txt)所在的文件夹路径
        percentile_clip (int): 截断极端异常值的百分位数，默认保留 99% 的数据用于画图，防止长尾效应压缩图表
    """
    aspect_ratios = []
    
    # 查找目录下所有的 txt 文件
    txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
    
    if not txt_files:
        print(f"错误：在 {label_dir} 中没有找到 .txt 文件。")
        return

    print(f"正在扫描 {len(txt_files)} 个标签文件...")

    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                # YOLO 格式: <class> <x_center> <y_center> <width> <height>
                if len(parts) >= 5:
                    try:
                        w = float(parts[3])
                        h = float(parts[4])
                        
                        if h > 0:  # 防止除以 0 的非法框
                            aspect_ratios.append(w / h)
                    except ValueError:
                        continue # 忽略无法解析的行

    if not aspect_ratios:
        print("未提取到有效的边界框数据。")
        return

    # 转换为 NumPy 数组便于统计
    ratios_array = np.array(aspect_ratios)
    
    # --- 打印统计数据 ---
    print("\n" + "="*30)
    print("长宽比 (w / h) 统计结果")
    print("="*30)
    print(f"总目标数 (Bounding Boxes): {len(ratios_array)}")
    print(f"最小长宽比: {np.min(ratios_array):.4f}")
    print(f"最大长宽比: {np.max(ratios_array):.4f}")
    print(f"平均长宽比 (Mean): {np.mean(ratios_array):.4f}")
    print(f"中位数 (Median): {np.median(ratios_array):.4f}")
    
    # 计算不同形状的比例
    tall_boxes = np.sum(ratios_array < 0.8)   # 偏竖直的长条形 (h > w)
    square_boxes = np.sum((ratios_array >= 0.8) & (ratios_array <= 1.2)) # 近似方形
    wide_boxes = np.sum(ratios_array > 1.2)   # 偏水平的扁平形 (w > h)
    
    print("\n目标形态分布:")
    print(f"- 纵向长条 (w/h < 0.8): {tall_boxes} 个 ({tall_boxes/len(ratios_array)*100:.2f}%)")
    print(f"- 近似方形 (0.8 <= w/h <= 1.2): {square_boxes} 个 ({square_boxes/len(ratios_array)*100:.2f}%)")
    print(f"- 横向扁平 (w/h > 1.2): {wide_boxes} 个 ({wide_boxes/len(ratios_array)*100:.2f}%)")

    # --- 绘制直方图 ---
    # 使用百分位数过滤极少数的极端长宽比（如因标注错误导致的 w/h > 20），以保证图表的可读性
    clip_val = np.percentile(ratios_array, percentile_clip)
    filtered_ratios = ratios_array[ratios_array <= clip_val]

    plt.figure(figsize=(10, 6))
    plt.hist(filtered_ratios, bins=60, color='#4A90E2', edgecolor='black', alpha=0.7)
    
    # 添加辅助线
    mean_val = np.mean(filtered_ratios)
    median_val = np.median(filtered_ratios)
    plt.axvline(mean_val, color='#E74C3C', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='#2ECC71', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')
    
    plt.title('Bounding Box Aspect Ratio (Width / Height) Distribution', fontsize=14)
    plt.xlabel('Aspect Ratio (w/h)', fontsize=12)
    plt.ylabel('Frequency (Number of Targets)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存并显示
    plt.savefig('aspect_ratio_distribution.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存为 'aspect_ratio_distribution.png'")
    plt.show()

# ==========================================
# 使用方法：将下面的路径替换为你数据集的标签路径
# ==========================================
if __name__ == '__main__':
    # 示例路径：假设你的VisDrone训练集标签在 ./datasets/VisDrone/labels/train 下
    TARGET_LABEL_DIR = '/3240410021/ultralytics-main/aaa/datasets/CODrone/train/labels' 
    
    if os.path.exists(TARGET_LABEL_DIR):
        analyze_yolo_aspect_ratios(TARGET_LABEL_DIR)
    else:
        print(f"请修改 TARGET_LABEL_DIR。当前路径 {TARGET_LABEL_DIR} 不存在。")