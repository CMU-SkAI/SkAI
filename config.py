import configparser

config = configparser.ConfigParser()
config.read('config.ini')

TRAIN_RATIO = config.getfloat('DEFAULT', 'TRAIN_RATIO', fallback=0.8)
VALIDATE_RATIO = config.getfloat('DEFAULT', 'VALIDATE_RATIO', fallback=0.1)
ROOT_ORIGIN_FOLDER= config.get('DEFAULT', 'ROOT_ORIGIN_FOLDER', fallback='skateboarding_deck_images')
ROOT_TRAIN_FOLDER = config.get('DEFAULT', 'ROOT_TRAIN_FOLDER', fallback='train_images')
ROOT_TEST_FOLDER = config.get('DEFAULT', 'ROOT_TEST_FOLDER', fallback='test_images')
ROOT_VALIDATE_FOLDER = config.get('DEFAULT', 'ROOT_VALIDATE_FOLDER', fallback='validate_images')
DATA_LOADER_NUM_WORKERS = config.getint('DEFAULT', 'DATA_LOADER_NUM_WORKERS', fallback=4)
ROOT_PREDICT_FOLDER = config.get('DEFAULT', 'ROOT_PREDICT_FOLDER', fallback='predict_images')
ROOT_MODELS_FOLDER = config.get('DEFAULT', 'ROOT_MODELS_FOLDER', fallback='models')

PRETRAINED_SIZE = config.getint('DEFAULT', 'PRETRAINED_SIZE', fallback=256)
PRETRAINED_MEANS = list(map(float, config.get('DEFAULT', 'PRETRAINED_MEANS', fallback='0.485,0.456,0.406').split(',')))
PRETRAINED_STDS = list(map(float, config.get('DEFAULT', 'PRETRAINED_STDS', fallback='0.229,0.224,0.225').split(',')))
BATCH_SIZE = config.getint('DEFAULT', 'BATCH_SIZE', fallback=256)
NUM_CLASSES = config.getint('DEFAULT', 'NUM_CLASSES', fallback=3)


ALEXNET_HIDDEN_UNITS = config.getint('ALEXNET', 'ALEXNET_HIDDEN_UNITS', fallback=4096)
ALEXNET_EPOCHS = config.getint('ALEXNET', 'ALEXNET_EPOCHS', fallback=10)
ALEXNET_MODEL_NAME = config.get('ALEXNET', 'ALEXNET_MODEL_NAME', fallback='alexnet_model_skai.pth')

RESNET_HIDDEN_UNITS = config.getint('RESNET', 'RESNET_HIDDEN_UNITS', fallback=512)
RESNET_EPOCHS = config.getint('RESNET', 'RESNET_EPOCHS', fallback=10)
RESNET_USE_CUDA = config.getboolean('RESNET', 'RESNET_USE_CUDA', fallback=True)
RESNET_MODEL_NAME = config.get('RESNET', 'RESNET_MODEL_NAME', fallback='resnet_model_skai.pth')
