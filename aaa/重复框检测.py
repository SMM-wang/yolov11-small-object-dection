import re
from itertools import combinations

def calculate_iou(box1, box2):
    """
    计算两个边界框的 IoU (交并比)
    box 格式: [x1, y1, x2, y2]
    """
    # 计算交集区域的坐标
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # 判断是否有交集
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    # 计算交集面积
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # 计算两个框的独立面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集面积
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def parse_detections(text):
    """
    解析输入文本，提取目标信息
    """
    # 正则表达式匹配目标格式
    pattern = r"目标\s*(\d+):\s*class=([\w\(\)]+),\s*x1=([\d\.]+),\s*y1=([\d\.]+),\s*x2=([\d\.]+),\s*y2=([\d\.]+),\s*conf=([\d\.]+)"
    
    detections = []
    for match in re.finditer(pattern, text):
        obj_id = match.group(1)
        cls = match.group(2)
        x1 = float(match.group(3))
        y1 = float(match.group(4))
        x2 = float(match.group(5))
        y2 = float(match.group(6))
        conf = float(match.group(7))
        
        detections.append({
            'id': obj_id,
            'class': cls,
            'box': [x1, y1, x2, y2],
            'conf': conf
        })
    return detections

def find_duplicates(detections, iou_threshold=0.7):
    """
    按类别分组并找出 IoU 大于阈值的重复框
    """
    # 按类别对目标进行分组
    class_groups = {}
    for det in detections:
        cls = det['class']
        if cls not in class_groups:
            class_groups[cls] = []
        class_groups[cls].append(det)

    duplicates = []
    
    # 在同类别内部进行两两对比
    for cls, items in class_groups.items():
        if len(items) < 2:
            continue
            
        # 使用 combinations 获取两两组合
        for obj1, obj2 in combinations(items, 2):
            iou = calculate_iou(obj1['box'], obj2['box'])
            if iou > iou_threshold:
                duplicates.append({
                    'class': cls,
                    'iou': iou,
                    'obj1': obj1,
                    'obj2': obj2
                })
                
    return duplicates

# ================= 测试代码 =================
if __name__ == "__main__":
    # 输入文本 (可以替换为您提供的长文本数据)
    data_text = """
  目标 1: class=3(car), x1=778.65, y1=617.29, x2=856.78, y2=701.36, conf=0.9404
  目标 2: class=3(car), x1=289.20, y1=542.21, x2=363.78, y2=616.04, conf=0.9249
  目标 3: class=3(car), x1=423.47, y1=475.70, x2=486.66, y2=547.62, conf=0.9243
  目标 4: class=3(car), x1=535.95, y1=560.15, x2=596.44, y2=628.03, conf=0.9236
  目标 5: class=3(car), x1=868.13, y1=622.33, x2=950.38, y2=706.51, conf=0.9142
  目标 6: class=3(car), x1=677.85, y1=597.83, x2=740.33, y2=679.84, conf=0.9106
  目标 7: class=3(car), x1=322.24, y1=440.18, x2=376.46, y2=487.56, conf=0.8894
  目标 8: class=3(car), x1=394.67, y1=425.86, x2=445.83, y2=471.44, conf=0.8861
  目标 9: class=3(car), x1=0.00, y1=672.65, x2=63.40, y2=764.12, conf=0.8838
  目标 10: class=3(car), x1=491.87, y1=376.69, x2=527.72, y2=413.37, conf=0.8802
  目标 11: class=3(car), x1=348.04, y1=472.52, x2=413.13, y2=534.65, conf=0.8788
  目标 12: class=3(car), x1=182.69, y1=653.20, x2=303.22, y2=763.37, conf=0.8784
  目标 13: class=3(car), x1=617.82, y1=433.87, x2=656.51, y2=475.44, conf=0.8783
  目标 14: class=3(car), x1=138.75, y1=604.14, x2=238.57, y2=696.95, conf=0.8739
  目标 15: class=3(car), x1=322.58, y1=383.73, x2=368.71, y2=423.60, conf=0.8686
  目标 16: class=3(car), x1=411.06, y1=350.29, x2=448.26, y2=383.14, conf=0.8685
  目标 17: class=3(car), x1=341.90, y1=326.93, x2=379.32, y2=355.58, conf=0.8641
  目标 18: class=3(car), x1=267.82, y1=378.24, x2=319.81, y2=417.86, conf=0.8587
  目标 19: class=3(car), x1=508.33, y1=337.10, x2=544.30, y2=370.26, conf=0.8549
  目标 20: class=3(car), x1=704.99, y1=412.09, x2=742.67, y2=449.35, conf=0.8529
  目标 21: class=3(car), x1=768.56, y1=388.20, x2=807.70, y2=422.23, conf=0.8514
  目标 22: class=3(car), x1=633.66, y1=368.40, x2=663.78, y2=397.70, conf=0.8448
  目标 23: class=3(car), x1=406.66, y1=316.80, x2=434.54, y2=341.90, conf=0.8329
  目标 24: class=3(car), x1=295.62, y1=645.71, x2=401.03, y2=765.00, conf=0.8198
  目标 25: class=3(car), x1=106.82, y1=546.28, x2=202.61, y2=626.23, conf=0.8117
  目标 26: class=3(car), x1=428.24, y1=296.60, x2=457.49, y2=319.82, conf=0.8029
  目标 27: class=0(pedestrian), x1=1047.43, y1=593.28, x2=1067.80, y2=643.43, conf=0.8002
  目标 28: class=3(car), x1=671.85, y1=315.00, x2=696.07, y2=340.01, conf=0.7972
  目标 29: class=3(car), x1=606.26, y1=262.49, x2=625.12, y2=279.48, conf=0.7923
  目标 30: class=3(car), x1=543.47, y1=268.96, x2=566.29, y2=289.49, conf=0.7922
  目标 31: class=0(pedestrian), x1=1102.13, y1=486.55, x2=1116.77, y2=521.93, conf=0.7677
  目标 32: class=0(pedestrian), x1=1108.61, y1=676.82, x2=1133.58, y2=735.79, conf=0.7567
  目标 33: class=3(car), x1=629.24, y1=242.03, x2=647.36, y2=258.61, conf=0.7555
  目标 34: class=0(pedestrian), x1=1281.49, y1=620.74, x2=1302.78, y2=672.83, conf=0.7478
  目标 35: class=3(car), x1=480.07, y1=279.63, x2=503.65, y2=299.95, conf=0.7475
  目标 36: class=3(car), x1=746.24, y1=326.71, x2=776.29, y2=352.17, conf=0.7261
  目标 37: class=0(pedestrian), x1=1138.30, y1=684.16, x2=1159.44, y2=736.85, conf=0.7175
  目标 38: class=0(pedestrian), x1=1185.31, y1=395.22, x2=1198.27, y2=425.42, conf=0.7121
  目标 39: class=0(pedestrian), x1=1135.44, y1=498.88, x2=1149.35, y2=538.08, conf=0.7116
  目标 40: class=3(car), x1=217.36, y1=461.14, x2=287.44, y2=516.02, conf=0.6918
  目标 41: class=0(pedestrian), x1=1270.33, y1=530.18, x2=1288.10, y2=578.43, conf=0.6903
  目标 42: class=8(bus), x1=473.44, y1=208.39, x2=508.17, y2=242.89, conf=0.6892
  目标 43: class=0(pedestrian), x1=1143.32, y1=388.10, x2=1156.94, y2=420.98, conf=0.6889
  目标 44: class=3(car), x1=656.47, y1=228.30, x2=673.10, y2=242.82, conf=0.6841
  目标 45: class=3(car), x1=573.19, y1=215.37, x2=588.42, y2=227.90, conf=0.6771
  目标 46: class=3(car), x1=712.88, y1=257.76, x2=738.20, y2=276.56, conf=0.6587
  目标 47: class=0(pedestrian), x1=1027.17, y1=459.26, x2=1040.35, y2=497.01, conf=0.6392
  目标 48: class=0(pedestrian), x1=1039.80, y1=458.79, x2=1051.82, y2=494.70, conf=0.6358
  目标 49: class=0(pedestrian), x1=1173.41, y1=331.41, x2=1184.39, y2=358.98, conf=0.6336
  目标 50: class=0(pedestrian), x1=1222.38, y1=522.43, x2=1237.91, y2=561.45, conf=0.6321
  目标 51: class=3(car), x1=495.24, y1=262.92, x2=516.82, y2=281.39, conf=0.6300
  目标 52: class=0(pedestrian), x1=1173.37, y1=512.96, x2=1188.14, y2=551.76, conf=0.6266
  目标 53: class=8(bus), x1=404.16, y1=245.87, x2=448.03, y2=298.07, conf=0.6222
  目标 54: class=0(pedestrian), x1=1263.02, y1=528.41, x2=1278.00, y2=573.98, conf=0.6192
  目标 55: class=0(pedestrian), x1=1177.09, y1=484.04, x2=1191.51, y2=516.99, conf=0.5978
  目标 56: class=3(car), x1=505.00, y1=229.09, x2=520.84, y2=242.91, conf=0.5957
  目标 57: class=0(pedestrian), x1=1194.27, y1=482.73, x2=1208.81, y2=518.24, conf=0.5923
  目标 58: class=2(bicycle), x1=1321.85, y1=602.79, x2=1350.82, y2=638.10, conf=0.5893
  目标 59: class=0(pedestrian), x1=1254.71, y1=355.08, x2=1265.34, y2=379.91, conf=0.5820
  目标 60: class=0(pedestrian), x1=1116.84, y1=322.96, x2=1125.73, y2=347.17, conf=0.5797
  目标 61: class=3(car), x1=589.36, y1=199.30, x2=602.64, y2=210.93, conf=0.5650
  目标 62: class=0(pedestrian), x1=1148.16, y1=339.88, x2=1158.15, y2=365.82, conf=0.5579
  目标 63: class=0(pedestrian), x1=1205.16, y1=331.25, x2=1215.90, y2=356.08, conf=0.5536
  目标 64: class=3(car), x1=573.73, y1=193.26, x2=585.15, y2=203.83, conf=0.5456
  目标 65: class=8(bus), x1=0.00, y1=423.34, x2=205.70, y2=610.89, conf=0.5416
  目标 66: class=3(car), x1=462.17, y1=302.41, x2=489.72, y2=327.38, conf=0.5366
  目标 67: class=3(car), x1=752.81, y1=348.29, x2=791.45, y2=379.07, conf=0.5241
  目标 68: class=0(pedestrian), x1=133.56, y1=338.57, x2=143.71, y2=365.94, conf=0.5234
  目标 69: class=3(car), x1=514.20, y1=219.33, x2=529.54, y2=232.44, conf=0.5181
  目标 70: class=0(pedestrian), x1=1244.33, y1=507.01, x2=1260.62, y2=541.89, conf=0.5069
  目标 71: class=0(pedestrian), x1=100.94, y1=416.79, x2=115.90, y2=451.91, conf=0.5063
  目标 72: class=0(pedestrian), x1=1202.56, y1=524.39, x2=1220.55, y2=570.06, conf=0.5015
  目标 73: class=3(car), x1=479.31, y1=251.54, x2=501.37, y2=268.16, conf=0.5001
  目标 74: class=0(pedestrian), x1=1023.09, y1=289.05, x2=1030.82, y2=310.66, conf=0.4953
  目标 75: class=0(pedestrian), x1=1199.73, y1=399.24, x2=1211.82, y2=426.21, conf=0.4953
  目标 76: class=4(van), x1=841.04, y1=544.26, x2=901.21, y2=616.75, conf=0.4825
  目标 77: class=3(car), x1=576.50, y1=179.84, x2=586.33, y2=188.88, conf=0.4777
  目标 78: class=3(car), x1=561.07, y1=184.96, x2=572.05, y2=194.21, conf=0.4446
  目标 79: class=9(motor), x1=1038.12, y1=540.81, x2=1085.39, y2=572.21, conf=0.4355
  目标 80: class=3(car), x1=638.52, y1=175.42, x2=648.01, y2=184.72, conf=0.4342
  目标 81: class=0(pedestrian), x1=123.27, y1=416.29, x2=135.83, y2=446.03, conf=0.4297
  目标 82: class=0(pedestrian), x1=1106.16, y1=365.07, x2=1118.21, y2=393.36, conf=0.4125
  目标 83: class=8(bus), x1=440.87, y1=245.35, x2=484.14, y2=296.13, conf=0.4117
  目标 84: class=3(car), x1=494.69, y1=237.23, x2=512.83, y2=251.16, conf=0.4053
  目标 85: class=3(car), x1=1124.68, y1=613.63, x2=1280.72, y2=686.92, conf=0.4031
  目标 86: class=0(pedestrian), x1=1228.31, y1=345.71, x2=1240.33, y2=373.62, conf=0.3984
  目标 87: class=3(car), x1=534.75, y1=199.21, x2=547.65, y2=210.17, conf=0.3909
  目标 88: class=3(car), x1=514.03, y1=243.37, x2=532.92, y2=260.40, conf=0.3829
  目标 89: class=3(car), x1=673.40, y1=211.04, x2=689.60, y2=224.23, conf=0.3815
  目标 90: class=0(pedestrian), x1=1211.55, y1=480.64, x2=1226.72, y2=518.47, conf=0.3783
  目标 91: class=0(pedestrian), x1=1084.49, y1=349.79, x2=1095.04, y2=377.25, conf=0.3771
  目标 92: class=0(pedestrian), x1=1240.69, y1=351.64, x2=1252.15, y2=376.40, conf=0.3739
  目标 93: class=3(car), x1=647.07, y1=186.48, x2=659.70, y2=198.22, conf=0.3597
  目标 94: class=0(pedestrian), x1=1138.10, y1=279.68, x2=1146.38, y2=302.20, conf=0.3596
  目标 95: class=0(pedestrian), x1=1120.75, y1=369.77, x2=1131.59, y2=398.80, conf=0.3387
  目标 96: class=2(bicycle), x1=1098.74, y1=617.96, x2=1134.19, y2=659.71, conf=0.3378
  目标 97: class=3(car), x1=533.66, y1=219.74, x2=549.89, y2=233.27, conf=0.3355
  目标 98: class=9(motor), x1=1099.71, y1=462.13, x2=1138.20, y2=491.52, conf=0.3322
  目标 99: class=1(people), x1=1053.28, y1=529.84, x2=1071.77, y2=566.56, conf=0.3302
  目标 100: class=1(people), x1=1309.67, y1=709.74, x2=1338.24, y2=762.42, conf=0.3189
  目标 101: class=3(car), x1=524.50, y1=234.62, x2=540.65, y2=249.18, conf=0.3156
  目标 102: class=3(car), x1=603.18, y1=168.66, x2=611.60, y2=176.13, conf=0.3152
  目标 103: class=0(pedestrian), x1=960.01, y1=257.94, x2=967.82, y2=277.07, conf=0.3115
  目标 104: class=3(car), x1=514.70, y1=208.80, x2=528.75, y2=220.03, conf=0.3100
  目标 105: class=2(bicycle), x1=1332.07, y1=582.15, x2=1360.00, y2=612.07, conf=0.3059
  目标 106: class=3(car), x1=542.92, y1=191.90, x2=555.39, y2=202.04, conf=0.2971
  目标 107: class=2(bicycle), x1=1099.71, y1=462.13, x2=1138.20, y2=491.52, conf=0.2948
  目标 108: class=3(car), x1=520.41, y1=192.26, x2=532.74, y2=202.19, conf=0.2888
  目标 109: class=9(motor), x1=1293.56, y1=733.02, x2=1343.70, y2=765.00, conf=0.2827
  目标 110: class=3(car), x1=542.48, y1=210.27, x2=557.95, y2=222.38, conf=0.2794
  目标 111: class=3(car), x1=556.95, y1=196.48, x2=570.38, y2=207.95, conf=0.2780
  目标 112: class=3(car), x1=597.06, y1=179.15, x2=606.33, y2=187.76, conf=0.2750
  目标 113: class=3(car), x1=544.13, y1=190.08, x2=555.47, y2=199.93, conf=0.2740
  目标 114: class=3(car), x1=462.66, y1=302.58, x2=489.58, y2=327.02, conf=0.2712
  目标 115: class=4(van), x1=295.62, y1=645.71, x2=401.03, y2=765.00, conf=0.2708
  目标 116: class=3(car), x1=647.85, y1=186.86, x2=660.48, y2=198.43, conf=0.2655
  目标 117: class=0(pedestrian), x1=1026.84, y1=378.76, x2=1037.71, y2=405.92, conf=0.2638
  目标 118: class=1(people), x1=742.60, y1=369.63, x2=751.91, y2=389.05, conf=0.2598
  目标 119: class=2(bicycle), x1=1293.56, y1=733.02, x2=1343.70, y2=765.00, conf=0.2591
  目标 120: class=3(car), x1=547.65, y1=203.55, x2=561.84, y2=215.20, conf=0.2569
  目标 121: class=3(car), x1=604.51, y1=156.05, x2=612.35, y2=162.69, conf=0.2559
  目标 122: class=0(pedestrian), x1=1171.27, y1=394.08, x2=1183.07, y2=423.87, conf=0.2521
  目标 123: class=0(pedestrian), x1=917.68, y1=242.95, x2=925.11, y2=261.79, conf=0.2503
    """

    # 1. 解析数据
    detections = parse_detections(data_text)
    print(f"共解析到 {len(detections)} 个目标。\n")

    # 2. 设置 IoU 阈值并查找重复项 (这里将阈值调为 0.7)
    IOU_THRESHOLD = 0.3
    dup_pairs = find_duplicates(detections, iou_threshold=IOU_THRESHOLD)

    # 3. 打印结果
    print(f"--- 发现 {len(dup_pairs)} 对 IoU > {IOU_THRESHOLD} 的同类别重复框 ---\n")
    for pair in dup_pairs:
        cls = pair['class']
        iou = pair['iou']
        o1 = pair['obj1']
        o2 = pair['obj2']
        
        print(f"类别: {cls} | IoU: {iou:.4f}")
        print(f"  -> 目标 {o1['id']} (conf={o1['conf']}): 坐标 [x1={o1['box'][0]}, y1={o1['box'][1]}, x2={o1['box'][2]}, y2={o1['box'][3]}]")
        print(f"  -> 目标 {o2['id']} (conf={o2['conf']}): 坐标 [x1={o2['box'][0]}, y1={o2['box'][1]}, x2={o2['box'][2]}, y2={o2['box'][3]}]\n")