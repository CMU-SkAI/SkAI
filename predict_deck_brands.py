import os
from PIL import Image
import torch
import numpy as np
from config import ROOT_PREDICT_FOLDER, PRETRAINED_SIZE, PRETRAINED_MEANS, PRETRAINED_STDS
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

def predict_deck_brands(model, device, class_to_idx):
    predict_image_paths = [os.path.join(ROOT_PREDICT_FOLDER, f) for f in os.listdir(ROOT_PREDICT_FOLDER) if os.path.isfile(os.path.join(ROOT_PREDICT_FOLDER, f))]
    img_list = [Image.open(img_path) for img_path in predict_image_paths]

    predict_batch = torch.stack([
        transforms.Compose([
                transforms.Resize(PRETRAINED_SIZE),
                transforms.CenterCrop(PRETRAINED_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean = PRETRAINED_MEANS, std = PRETRAINED_STDS)
        ])(img).to(device) for img in img_list ])

    pred_logits_tensor = model(predict_batch)
    pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
    
    # Define the 2D NumPy array
    arr = np.array(pred_probs)

    fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
    for i, img in enumerate(img_list):
        ax = axs[i]
        ax.axis('off')
        ax.set_title("{:.2f}% {}".format( np.amax(arr[i, :], axis=0)*100 , list(class_to_idx.keys())[np.argmax(arr[i, :], axis=0)] ))
        ax.imshow(img)