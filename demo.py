import torch
from ultralytics import YOLO

yolo = YOLO("runs/detect/visdrone2019结果/ALL/weights/best.pt")
model = yolo.model

for name, m in model.named_modules():
    if m.__class__.__name__ == "C2PSA":
        print(f"\n=== {name} (C2PSA) ===")
        print(f"  self.c = {m.c}")
        print(f"  cv1.conv: in={m.cv1.conv.in_channels} out={m.cv1.conv.out_channels}")
        print(f"  cv2.conv: in={m.cv2.conv.in_channels} out={m.cv2.conv.out_channels}")
        for bname, bm in m.named_modules():
            if isinstance(bm, torch.nn.Conv2d):
                print(f"  {bname}: in={bm.in_channels} out={bm.out_channels} groups={bm.groups}")