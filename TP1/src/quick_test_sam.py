import numpy as np
import cv2
from pathlib import Path
from sam_utils import load_sam_predictor, predict_mask_from_box

img_path = next(Path("TP1/data/images").glob("*.jpg"))
bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

ckpt = "TP1/models/sam_vit_h_4b8939.pth" 
pred = load_sam_predictor(ckpt, model_type="vit_h")

box = np.array([50, 50, 250, 250], dtype=np.int32)

mask, score = predict_mask_from_box(pred, rgb, box, multimask=True)

print("img", rgb.shape)
print("mask", mask.shape)
print("score", score)
print("mask_sum", mask.sum())

