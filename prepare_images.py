import os
import shutil
import random
from PIL import Image, ImageOps
from config import ROOT_TRAIN_FOLDER, ROOT_TEST_FOLDER, ROOT_VALIDATE_FOLDER, ROOT_ORIGIN_FOLDER, TRAIN_RATIO, VALIDATE_IN_NON_TRAIN_RATIO

def process_images(brand_path, image, src_root_folder, dst_root_folder):
    image_src = os.path.join(src_root_folder, brand_path, image)
    image_dst = os.path.join(dst_root_folder, brand_path, image) 
    
    img = Image.open(image_src)
    img = resize_with_padding(img, (256, 256))
    if img.mode in ("RGBA", "P"): 
        img = img.convert("RGB")
    img.save(image_dst) 

def prepare_images():
    if os.path.exists(ROOT_TRAIN_FOLDER):
        shutil.rmtree(ROOT_TRAIN_FOLDER) 
    
    if os.path.exists(ROOT_TEST_FOLDER):
        shutil.rmtree(ROOT_TEST_FOLDER) 
    
    brands = [d for d in os.listdir(ROOT_ORIGIN_FOLDER) if os.path.isdir(os.path.join(ROOT_ORIGIN_FOLDER, d))]
    
    for brand in brands:
        images = [f for f in os.listdir(os.path.join(ROOT_ORIGIN_FOLDER, brand)) if os.path.isfile(os.path.join(os.path.join(ROOT_ORIGIN_FOLDER, brand), f))]
        # remove all non image files
        for image in images:
            if not image.endswith('.jpg') and not image.endswith('.png') and not image.endswith('.jpeg') and not image.endswith('.gif'): 
                images.remove(image)

        random.shuffle(images)
        n_train = int(len(images) * TRAIN_RATIO) 
        train_images = images[:n_train]
        non_train_images = images[n_train:]
        
        random.shuffle(non_train_images)
        n_validate = int(len(non_train_images) * VALIDATE_IN_NON_TRAIN_RATIO)
        validate_images = non_train_images[:n_validate]
        test_images = non_train_images[n_validate:]
        
        os.makedirs(os.path.join(ROOT_TRAIN_FOLDER, brand), exist_ok = True)
        os.makedirs(os.path.join(ROOT_TEST_FOLDER, brand), exist_ok = True)
        os.makedirs(os.path.join(ROOT_VALIDATE_FOLDER, brand), exist_ok = True)

        for image in train_images:
            process_images(brand, image, ROOT_ORIGIN_FOLDER, ROOT_TRAIN_FOLDER)

        for image in test_images:
            process_images(brand, image, ROOT_ORIGIN_FOLDER, ROOT_TEST_FOLDER)
        
        for image in validate_images:
            process_images(brand, image, ROOT_ORIGIN_FOLDER, ROOT_VALIDATE_FOLDER)
                
def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding, fill=(255,255,255))

if __name__ == '__main__':
    prepare_images()