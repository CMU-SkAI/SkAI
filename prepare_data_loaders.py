
from config import PRETRAINED_SIZE, PRETRAINED_MEANS, PRETRAINED_STDS, BATCH_SIZE, DATA_LOADER_NUM_WORKERS
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def prepare_data_loaders():
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(PRETRAINED_SIZE),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(mean = PRETRAINED_MEANS, 
                                    std = PRETRAINED_STDS),
            transforms.RandomCrop(PRETRAINED_SIZE, padding = 10)
        ]),
        'val': transforms.Compose([
            transforms.Resize(PRETRAINED_SIZE),
            transforms.CenterCrop(PRETRAINED_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean = PRETRAINED_MEANS, 
                                    std = PRETRAINED_STDS)
        ]),
        'test': transforms.Compose([
            transforms.Resize(PRETRAINED_SIZE),
            transforms.CenterCrop(PRETRAINED_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean = PRETRAINED_MEANS, 
                                    std = PRETRAINED_STDS)
        ])
    }
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(x + '_images'), data_transforms[x]) for x in ['train', 'val', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=DATA_LOADER_NUM_WORKERS) for x in ['train', 'val', 'test']}

    return dataloaders, image_datasets['train'].class_to_idx