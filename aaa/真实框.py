import cv2
import os


def draw_yolo_box(img_path, txt_path, class_names=None, save_dir=None):
    """
    读取图片和YOLO格式的txt，画出真实框，并保存为 "原名_true.jpg"
    :param img_path: 图片路径
    :param txt_path: 对应的标签txt路径
    :param class_names: 类别名称列表
    :param save_dir: (可选) 指定保存的文件夹路径。如果不填，默认保存在原图片所在文件夹。
    """

    # 1. 读取图像
    if not os.path.exists(img_path):
        print(f"错误: 找不到图片路径 {img_path}")
        return

    image = cv2.imread(img_path)
    h, w, c = image.shape

    # 2. 读取 txt 文件
    if not os.path.exists(txt_path):
        print(f"错误: 找不到txt路径 {txt_path}")
        return

    # 使用 utf-8 编码读取，防止中文乱码
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"正在处理: {img_path}")

    # 3. 遍历每一行数据
    for line in lines:
        data = line.strip().split()
        if len(data) < 5: continue

        class_id = int(data[0])
        x_center = float(data[1])
        y_center = float(data[2])
        width = float(data[3])
        height = float(data[4])

        # --- 坐标转换 ---
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        # 4. 画框
        color_map = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]
        color = color_map[class_id % len(color_map)]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # # 5. 写标签
        # if class_names and class_id < len(class_names):
        #     label = f"{class_names[class_id]}"
        # else:
        #     label = f"ID: {class_id}"
        #
        # (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        # cv2.rectangle(image, (x1, y1 - 20), (x1 + tw, y1), color, -1)
        # cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.6, (255, 255, 255), 1)

    # --- 6. 保存逻辑 (修改部分) ---

    # 获取文件名和后缀 (例如: '5' 和 '.jpg')
    base_name = os.path.basename(img_path)
    name_only, extension = os.path.splitext(base_name)

    # 构造新名字: 5_true.jpg
    new_filename = f"{name_only}_true{extension}"

    # 确定保存路径
    if save_dir:
        # 如果指定了文件夹，确保文件夹存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"已新建保存文件夹: {save_dir}")
        output_path = os.path.join(save_dir, new_filename)
    else:
        # 没指定文件夹，就默认保存在原图片的同级目录
        output_path = os.path.join(os.path.dirname(img_path), new_filename)

    # 保存图片
    cv2.imwrite(output_path, image)
    print(f"✅ 图片已保存至: {output_path}")

    # (可选) 显示结果，不想看弹窗可以注释掉下面两行
    # cv2.imshow("Result", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# ==========================================
#              运行配置
# ==========================================

# 1. 输入路径
img_file = r"./test_picture/drone6.jpg"
txt_file = r"./test_picture/drone6.txt"

# 2. 类别
# classes = ["broken", "insulator", "pollution-flashover"]
classes = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]


# 3. 指定保存文件夹 (例如保存到桌面下的 'result' 文件夹)
# 如果想直接保存在原文件夹，这里填 None 即可
output_folder = r"./test_picture"

# 运行
draw_yolo_box(img_file, txt_file, classes, save_dir=output_folder)