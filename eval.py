import os
import random
from tqdm import tqdm

import numpy as np
import cv2
from sklearn.metrics import top_k_accuracy_score

import torch
import torchvision.transforms as T

from model.vgg16 import VGG16


DATA_FOLDER = "data/test"
MODEL_PATH = "runs/weights/last.pth"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

IMAGE_PREPROCESS = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def read_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, dsize=(224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = IMAGE_PREPROCESS(img)
    img = img.unsqueeze(0).to(DEVICE)
    return img

LABEL_LIST = ["cat", "dog", "snake"]
def post_process(output):
    pred = output.softmax(dim=-1)[0].detach().cpu().numpy()
    pred_id = pred.argmax()
    lb = LABEL_LIST[pred_id]
    conf = pred[pred_id]
    return lb, conf


data_folder_list = list(map(lambda lb: os.path.join(DATA_FOLDER, lb), LABEL_LIST))
data_name_list = list(map(lambda f: sorted(os.listdir(f)), data_folder_list))
data_path_list = [list(map(lambda n: os.path.join(folder, n), name_list)) for folder, name_list in zip(data_folder_list, data_name_list)]
data_path_list = np.array(data_path_list)
data_path_list = data_path_list.reshape(-1)
# np.random.shuffle(data_path_list)

model = VGG16(3, 3, True)
ckpt = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(DEVICE)
model.eval()

# gt_list, pred_lb_list = [], []
all_gt_ids, all_output_scores = [], []
for path in tqdm(data_path_list):
    gt = LABEL_LIST.index(path.split('/')[-2])
    all_gt_ids.append(gt)
    # gt_list.append(path.split('/')[-2])

    img_tensor = read_image(path)
    output = model(img_tensor)
    # pred_lb, pred_conf = post_process(output)
    # pred_lb_list.append(pred_lb)
    all_output_scores.append(output.softmax(dim=-1)[0].detach().cpu().numpy())
    
all_output_scores = np.stack(all_output_scores, axis=0)
all_gt_ids = np.array(all_gt_ids)

acc_top1 = top_k_accuracy_score(all_gt_ids, all_output_scores, k=1)
acc_top5 = top_k_accuracy_score(all_gt_ids, all_output_scores, k=2)

print(f"EVALUATION ACCURACY RESULTS: TOP-1={acc_top1*100}% --- TOP-2={acc_top5*100}%")
