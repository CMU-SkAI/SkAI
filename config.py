import configparser

config = configparser.ConfigParser()
config.read('config.ini')

TRAIN_RATIO = config.getfloat('DEFAULT', 'TRAIN_RATIO', fallback=0.8)
VALIDATE_RATIO = config.getfloat('DEFAULT', 'VALIDATE_RATIO', fallback=0.1)
ROOT_ORIGIN_FOLDER= config.get('DEFAULT', 'ROOT_ORIGIN_FOLDER', fallback='skateboarding_deck_images')
ROOT_TRAIN_FOLDER = config.get('DEFAULT', 'ROOT_TRAIN_FOLDER', fallback='train_images')
ROOT_TEST_FOLDER = config.get('DEFAULT', 'ROOT_TEST_FOLDER', fallback='test_images')
ROOT_VALIDATE_FOLDER = config.get('DEFAULT', 'ROOT_VALIDATE_FOLDER', fallback='validate_images')

PRETRAINED_SIZE = config.getint('DEFAULT', 'PRETRAINED_SIZE', fallback=256)
PRETRAINED_MEANS = list(map(float, config.get('DEFAULT', 'PRETRAINED_MEANS', fallback='0.485,0.456,0.406').split(',')))
PRETRAINED_STDS = list(map(float, config.get('DEFAULT', 'PRETRAINED_STDS', fallback='0.229,0.224,0.225').split(',')))
BATCH_SIZE = config.getint('DEFAULT', 'BATCH_SIZE', fallback=256)
NUM_CLASSES = config.getint('DEFAULT', 'NUM_CLASSES', fallback=3)


HIDDEN_UNITS = config.getint('ALEXNET', 'HIDDEN_UNITS', fallback=4096)