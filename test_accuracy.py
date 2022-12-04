import torch
from sklearn.metrics import confusion_matrix as conf_mat
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from itertools import chain

def test_accuracy(model, dataloaders, device, type_name, class_to_idx):
    predictions = []
    label_data = []
    test_accuracy = 0
    test_len = 0

    for images, labels in dataloaders['test']:
        # Move images & labels to the GPU
        images, labels = images.to(device), labels.to(device)
        
        log_ps = model(images)
        ps = torch.exp(log_ps)
        top_ps, top_class = ps.topk(1, dim=1)

        # Save the predictions and labels for the test data
        label_data.extend(labels.cpu().data.numpy().tolist())
        predictions.extend(list(chain.from_iterable(top_class.cpu().data.numpy().tolist())))
        
        matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
        test_batch_accuracy = matches.mean()
        
        test_accuracy += test_batch_accuracy.item()  
        test_len += 1
    
    print(type_name + ' Test Accuracy:', test_accuracy / test_len)

    cf = conf_mat(label_data,predictions)
    ncf = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10,8))
    sn.heatmap(ncf, annot=True, fmt='.2f', xticklabels=class_to_idx, yticklabels=class_to_idx)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    