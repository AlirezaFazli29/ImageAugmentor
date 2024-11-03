import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_image(image_path: str):
    image = cv2.imread(image_path)
    return image

def convert_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def convert_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def read_label(label_pah: str):
    with open(label_pah, 'r') as file:
        label = file.read().strip()
    return label

def show_image(image, turn_grey=False,cmap=None):
    if turn_grey==True: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.show()

def yolo_to_pixel_coords(label, image_width, image_height):
    labels = label.split("\n")
    out = [None] * len(labels)
    class_ids = [None] * len(labels)
    for i, l in enumerate(labels):
        class_id, x_center, y_center, width, height = map(float, l.split())
        x_center_pixel = x_center * image_width
        y_center_pixel = y_center * image_height
        width_pixel = width * image_width
        height_pixel = height * image_height
        x_min = int(x_center_pixel - width_pixel / 2)
        y_min = int(y_center_pixel - height_pixel / 2)
        x_max = int(x_center_pixel + width_pixel / 2)
        y_max = int(y_center_pixel + height_pixel / 2)
        out[i] = [[x_min, y_min], [x_max, y_max]]
        class_ids[i] = class_id
    return np.array(class_ids), np.array(out)

def pixel_coords_to_yolo(pixel_coords, image_width, image_height, class_ids):
    yolo_format_boxes = []
    for i, box in enumerate(pixel_coords):
        (x_min, y_min), (x_max, y_max) = box
        x_center = (x_min + x_max) / 2 / image_width
        y_center = (y_min + y_max) / 2 / image_height
        width = abs((x_max - x_min) / image_width)
        height = abs((y_max - y_min) / image_height)
        class_id = class_ids[i]
        yolo_format_boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return yolo_format_boxes

def flatten_bounding_boxes(coords_array: np.ndarray):
    return coords_array.reshape(-1, coords_array.shape[-1])

def reshape_to_bounding_boxes(flattened_coords):
    if flattened_coords.shape[0] % 2 != 0:
        raise ValueError("Number of coordinates must be even to reshape into bounding boxes.")
    return flattened_coords.reshape(-1, 2, flattened_coords.shape[-1])