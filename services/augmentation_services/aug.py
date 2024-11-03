import albumentations as A
import services.data_handler.utils as dtils
import numpy as np
import os
from tqdm import tqdm
from cv2 import imwrite

class ImageAugmentor():

    def __init__(self, compose=None, keypoint_format='xy', remove_invisible=False):
        if compose==None:
            self.augmentations = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussianBlur(p=0.2),
                A.MotionBlur(p=0.2),
                A.GaussNoise(p=0.2),
                A.HueSaturationValue(p=0.2),
                A.RGBShift(p=0.2),
                A.ChannelShuffle(p=0.2),
                A.CLAHE(p=0.2),
                A.Equalize(p=0.2),
                A.FancyPCA(p=0.2),
                A.RandomGamma(p=0.2),
                A.ColorJitter(p=0.2),
                A.RandomFog(p=0.2)
            ], keypoint_params=A.KeypointParams(format=keypoint_format, remove_invisible=remove_invisible))
        else: self.augmentations = compose

    def augmet_image(self, image: np.ndarray, keypoints: np.ndarray):
        augmented = self.augmentations(image=image, keypoints=keypoints)
        return augmented
    
    def create_image_variants(self, image: np.ndarray, keypoints: np.ndarray, num_variants: int=10):
        images = [None] * num_variants
        keypoints_ = [None] * num_variants
        for i in range(num_variants):
            temp = self.augmet_image(image, keypoints)
            images[i] = temp["image"]
            keypoints_[i] = temp["keypoints"]
        return images, keypoints_
    
    def augment_and_save(self, image: np.ndarray, labels: np.ndarray, output_dir: str, num_variants: int=10, filename: str="test"):
        os.makedirs(output_dir, exist_ok=True)
        output_img_dir = os.path.join(output_dir, "images")
        output_label_dir = os.path.join(output_dir, "labels")
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)
        image_height, image_width = image.shape[:2]
        class_ids, keypoints = dtils.yolo_to_pixel_coords(labels, image_width, image_height)
        keypoints = dtils.flatten_bounding_boxes(keypoints)
        created_images, created_labels = self.create_image_variants(image, keypoints, num_variants)
        for i, lbl in enumerate(created_labels):
            created_labels[i] = dtils.reshape_to_bounding_boxes(lbl)
            created_labels[i] = dtils.pixel_coords_to_yolo(created_labels[i], image_width, image_height, class_ids)
        for i, (aug_image, aug_keypoints) in enumerate(zip(created_images, created_labels)):
            image_filename = f"{filename}_{i + 1}.jpg"
            output_image_path = os.path.join(output_img_dir, image_filename)
            aug_image_bgr = dtils.convert_to_bgr(aug_image)
            imwrite(output_image_path, aug_image_bgr)
            label_filename = f"{filename}_{i + 1}.txt"
            output_label_path = os.path.join(output_label_dir, label_filename)
            with open(output_label_path, "w") as label_file:
                label_file.write("\n".join(aug_keypoints))

    def augment_from_dir(self, image_dir: str, label_dir: str, output_dir: str, num_variant: int=10):
        image_names = os.listdir(image_dir)
        progress_bar = tqdm(image_names, total=len(image_names), desc= f"Augmentation Process", leave=True)
        for img_name in progress_bar:
            name_list = img_name.split('.')
            image_format = name_list[-1]
            base_filename = '.'.join(name_list[:-1])
            img = dtils.read_image(os.path.join(image_dir, f"{base_filename}.{image_format}"))
            img = dtils.convert_to_rgb(img)
            lbl = dtils.read_label(os.path.join(label_dir,f"{base_filename}.txt"))
            if not lbl.strip():
                print(f"Skipping {img_name} as it has an empty label file.")
                continue
            label_lines = lbl.split("\n")
            valid_format = True
            for line in label_lines:
                parts = line.split()
                if len(parts) != 5:
                    print(f"Skipping {img_name} as it has an invalid label format.")
                    valid_format = False
                    break
            if not valid_format:
                continue
            self.augment_and_save(
                image=img, labels=lbl,
                output_dir=output_dir,
                num_variants=num_variant,
                filename=base_filename
            )
        print(f"All images are successfully augmented and save to the {output_dir} directory")