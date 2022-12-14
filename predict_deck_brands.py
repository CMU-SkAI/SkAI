import os
from PIL import Image
import torch
import numpy as np
from config import ROOT_PREDICT_FOLDER, PRETRAINED_SIZE, PRETRAINED_MEANS, PRETRAINED_STDS
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from prepare_images import resize_with_padding 

def predict_deck_brands(model_name, model, device, class_to_idx):
    predict_image_paths = [os.path.join(ROOT_PREDICT_FOLDER, f) for f in os.listdir(ROOT_PREDICT_FOLDER) if os.path.isfile(os.path.join(ROOT_PREDICT_FOLDER, f))]
    # remove all non image files
    for image_path in predict_image_paths:
        if not image_path.endswith('.jpg') and not image_path.endswith('.png') and not image_path.endswith('.jpeg') and not image_path.endswith('.gif'): 
            predict_image_paths.remove(image_path)
             
    img_list = []
    for img_path in predict_image_paths:
        img = Image.open(img_path)
        img = resize_with_padding(img, (256, 256))
        if img.mode in ("RGBA", "P"): 
            img = img.convert("RGB")
        img_list.append(img)

    transform_arr = []
    for idx, img in enumerate(img_list):
        try:
            temp = transforms.Compose([
                    transforms.Resize(PRETRAINED_SIZE),
                    transforms.CenterCrop(PRETRAINED_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = PRETRAINED_MEANS, std = PRETRAINED_STDS)
            ])(img).to(device)
        except:
            print('Color channel is not 3 in this image: ' + predict_image_paths[idx])
            img_list.remove(img)
            predict_image_paths.remove(predict_image_paths[idx])
        transform_arr.append(temp)

    predict_batch = torch.stack(transform_arr)

    # Add a new dimension to the tensor using numpy.expand_dims()
    # predict_batch = np.expand_dims(predict_batch, axis=1)

    pred_logits_tensor = model(predict_batch)
    pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
    
    # Define the 2D NumPy array
    arr = np.array(pred_probs)

    fig, axs = plt.subplots(int(math.ceil( len(img_list) / 3 )), 3, figsize=(20, 5 * int(math.ceil( len(img_list) / 3 )) ))
    correct_counter = 0
    for i, ax in enumerate(axs.flat):
        if i < len(img_list):
            ax.axis('off')
            ax.set_title("{:.2f}% [{}]\n{}".format( np.amax(arr[i, :], axis=0)*100 , list(class_to_idx.keys())[np.argmax(arr[i, :], axis=0)], os.path.basename(predict_image_paths[i]) ))
            ax.imshow(img_list[i])
            # lowercase the filename and the brand name, then compare them
            if list(class_to_idx.keys())[np.argmax(arr[i, :], axis=0)].lower() in os.path.basename(predict_image_paths[i]).lower():
                correct_counter += 1
        else:
            # remove this ex from the diagram
            ax.axis('off')
    fig.suptitle('[' + model_name + '] Correctly predicted: ' + str(correct_counter) + ' out of ' + str(len(img_list)) + ' images, Accuracy: ' + str(round(correct_counter / len(img_list) * 100, 2)) + '%')
