
from config import ROOT_TRAIN_FOLDER, ROOT_VALIDATE_FOLDER, ROOT_TEST_FOLDER, PRETRAINED_SIZE, PRETRAINED_MEANS, PRETRAINED_STDS, BATCH_SIZE
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def prepare_data_loaders():
    train_transforms = transforms.Compose([
                            transforms.Resize(PRETRAINED_SIZE),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomRotation(5),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = PRETRAINED_MEANS, 
                                                    std = PRETRAINED_STDS),
                            transforms.RandomCrop(PRETRAINED_SIZE, padding = 10)
                        ])

    validate_transforms = transforms.Compose([
                            transforms.Resize(PRETRAINED_SIZE),
                            transforms.CenterCrop(PRETRAINED_SIZE),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = PRETRAINED_MEANS, 
                                                    std = PRETRAINED_STDS)
                        ])
    
    test_transforms = transforms.Compose([
                            transforms.Resize(PRETRAINED_SIZE),
                            transforms.CenterCrop(PRETRAINED_SIZE),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = PRETRAINED_MEANS, 
                                                    std = PRETRAINED_STDS)
                        ])
    
    train_dataset = datasets.ImageFolder(ROOT_TRAIN_FOLDER, transform = train_transforms)
    validate_dataset = datasets.ImageFolder(ROOT_VALIDATE_FOLDER, transform = validate_transforms)
    test_dataset = datasets.ImageFolder(ROOT_TEST_FOLDER, transform = test_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    validate_dataloader = DataLoader(validate_dataset, batch_size = BATCH_SIZE, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)

    return {
        'train_dataloader': train_dataloader,
        'validate_dataloader': validate_dataloader,
        'test_dataloader': test_dataloader,
        'len_train': len(train_dataset),
        'len_validate': len(validate_dataset),
        'len_test': len(test_dataset)
    }