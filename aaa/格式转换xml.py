import xml.etree.ElementTree as ET
import os
from glob import glob
from tqdm import tqdm

# ================= 核心配置 (请仔细核对) =================
# 1. 这里的名字必须和 XML <name> 标签里的完全一样！
#    根据你刚才发的 XML，我已经填了一个 "missing_hole"。
#    请把剩下的 5 个类别名字也填进去（顺序不要变）。
classes = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper"
]

# 2. 数据集根目录
dataset_root = r"C:\workspace\python\ultralytics-main\datasets\PCB_DATASET"


# ========================================================

def convert(size, box):
    """ 坐标归一化转换 """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def check_classes_first():
    """ 预检查：扫描所有XML，看看有哪些类别，防止写错导致文件为空 """
    print("正在检查 XML 中的类别名称...")
    found_classes = set()
    xml_files = glob(os.path.join(dataset_root, "*", "labels", "*.xml"))

    if not xml_files:
        print("错误：在 labels 文件夹里没找到 XML 文件！请检查 dataset_split 目录结构。")
        return False

    # 随机抽查 100 个文件，或者检查全部
    for xf in xml_files[:200]:
        try:
            tree = ET.parse(xf)
            root = tree.getroot()
            for obj in root.iter('object'):
                found_classes.add(obj.find('name').text)
        except:
            pass

    print(f"扫描到的类别名称有: {found_classes}")
    print(f"你代码里填写的类别有: {set(classes)}")

    # 检查是否有漏网之鱼
    diff = found_classes - set(classes)
    if diff:
        print(f"\n⚠️  严重警告：XML里发现了代码里没写的类别: {diff}")
        print("这会导致生成的 TXT 为空！请立即把这些名字加到 classes 列表里！")
        cmd = input("是否继续运行？(y/n): ")
        if cmd.lower() != 'y':
            return False
    return True


def convert_annotation():
    if not check_classes_first():
        return

    sets = ['train', 'val', 'test']

    for image_set in sets:
        labels_path = os.path.join(dataset_root, image_set, 'labels')
        if not os.path.exists(labels_path):
            continue

        xml_files = glob(os.path.join(labels_path, "*.xml"))
        print(f"\n正在处理 {image_set} 集，共 {len(xml_files)} 个文件...")

        count_success = 0

        for xml_file in tqdm(xml_files):
            file_id = os.path.splitext(os.path.basename(xml_file))[0]
            txt_file = os.path.join(labels_path, file_id + ".txt")

            try:
                in_file = open(xml_file, encoding='utf-8')
                tree = ET.parse(in_file)
                root = tree.getroot()

                size = root.find('size')
                w = int(size.find('width').text)
                h = int(size.find('height').text)

                if w == 0 or h == 0:
                    in_file.close()
                    continue

                out_file = open(txt_file, 'w', encoding='utf-8')
                has_object = False

                for obj in root.iter('object'):
                    cls = obj.find('name').text

                    if cls not in classes:
                        continue  # 遇到不在列表里的类别，跳过

                    cls_id = classes.index(cls)
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                         float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                    bb = convert((w, h), b)

                    out_file.write(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")
                    has_object = True

                out_file.close()
                in_file.close()

                # === 关键修改：清理文件 ===
                # 只有成功写入了内容，才删除 xml，防止误删
                if has_object:
                    os.remove(xml_file)  # 删除 XML，解决“混在一起”的问题
                    count_success += 1
                else:
                    # 如果 TXT 是空的（因为没有匹配的类别），把空 TXT 删了，保留 XML 排查
                    os.remove(txt_file)

            except Exception as e:
                print(f"处理 {xml_file} 时出错: {e}")

        print(f"完成！成功转换并删除了 {count_success} 个 XML 文件。")


if __name__ == "__main__":
    convert_annotation()