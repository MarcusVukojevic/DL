


import torch
from torchvision.transforms import v2
from PIL import Image
import matplotlib.pyplot as plt
from augmix import _augmix_aug
from torchvision import transforms
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import random
from PIL import Image
import concurrent.futures

class ImageAugmentor:
    def __init__(self, augmentations=None):
        if augmentations is None:
            self.augmentations = v2.Compose([
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandomRotation(degrees=45),
                v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            ])
        else:
            self.augmentations = augmentations

    def apply_augmentations(self, image_path, aug, n_augmentations):
        # Carica e converte l'immagine in RGB
        image = Image.open(image_path).convert("RGB")
        
        # Definisci una funzione helper per eseguire l'augmentazione
        def augment_image(_):
            if aug:
                return transforms.ToPILImage()(_augmix_aug(image))
            else:
                return self.augmentations(image)
        
        # Usa ThreadPoolExecutor per eseguire le augmentazioni in parallelo
        with concurrent.futures.ThreadPoolExecutor() as executor:
            augmented_images = list(executor.map(augment_image, range(n_augmentations)))
    
        return augmented_images
    


    def apply_all_augmentations(self, image_path):
        image = Image.open(image_path).convert('RGB')
        augmentations = []

        def apply_augmentation(func, *args):
            return func(*args)

        tasks = [
            (random_rotation, image, 30),
            (random_crop, image, (int(image.width * 0.8), int(image.height * 0.8))),
            (random_zoom, image, 0.8, 1.2),
            (random_shift, image, 10, 10),
            (shear_image, image, 0.2),
            (adjust_brightness, image, 1.5),
            (adjust_contrast, image, 1.5),
            (adjust_saturation, image, 1.5),
            (adjust_hue, image, 50),
            (add_noise, image, 25),
            (blur_image, image, 2),
            (sharpen_image, image, 2),
            (grayscale_image, image),
            (cutout_image, image, 50),
            (flip_image_horizontal, image),
            (flip_image_vertical, image),
            (rotate_image, image, 45),
            (crop_image, image, (10, 10, image.width-10, image.height-10)),
            (zoom_image, image, 1.1),
            (shift_image, image, 5, 5),
            (random_rotation, image, 30),
            (random_crop, image, (int(image.width * 0.8), int(image.height * 0.8))),
            (random_zoom, image, 0.8, 1.2),
            (random_shift, image, 10, 10),
            (shear_image, image, 0.2),
            (adjust_brightness, image, 1.5),
            (adjust_contrast, image, 1.5),
            (adjust_saturation, image, 1.5),
            (adjust_hue, image, 50),
            (add_noise, image, 25),
            (blur_image, image, 2),
            (sharpen_image, image, 2),
            (grayscale_image, image),
            (cutout_image, image, 50),
            (flip_image_horizontal, image),
            (flip_image_vertical, image),
            (rotate_image, image, 45),
            (crop_image, image, (10, 10, image.width-10, image.height-10)),
            (zoom_image, image, 1.1),
            (shift_image, image, 5, 5),
            (random_rotation, image, 30),
            (random_crop, image, (int(image.width * 0.8), int(image.height * 0.8))),
            (random_zoom, image, 0.8, 1.2),
            (random_shift, image, 10, 10),
            (shear_image, image, 0.2),
            (adjust_brightness, image, 1.5),
            (adjust_contrast, image, 1.5),
            (adjust_saturation, image, 1.5),
            (adjust_hue, image, 50),
            (add_noise, image, 25),
            (blur_image, image, 2),
            (sharpen_image, image, 2),
            (grayscale_image, image),
            (cutout_image, image, 50),
            (flip_image_horizontal, image),
            (flip_image_vertical, image),
            (rotate_image, image, 45),
            (crop_image, image, (10, 10, image.width-10, image.height-10)),
            (zoom_image, image, 1.1),
            (shift_image, image, 5, 5)
        ]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [executor.submit(apply_augmentation, task[0], *task[1:]) for task in tasks]
            for future in concurrent.futures.as_completed(results):
                augmentations.append(future.result())

        return augmentations


    def show_images(self, original_image, augmented_images):
        fig, axes = plt.subplots(1, len(augmented_images) + 1, figsize=(15, 5))
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        try:
            for i, aug_image in enumerate(augmented_images):
                axes[i + 1].imshow(transforms.ToPILImage()(aug_image))
                axes[i + 1].set_title(f"Augmented Image {i+1}")
                axes[i + 1].axis("off")
        except:
            for i, aug_image in enumerate(augmented_images):
                axes[i + 1].imshow(aug_image)
                axes[i + 1].set_title(f"Augmented Image {i+1}")
                axes[i + 1].axis("off")
        
        plt.show()


# Definizione delle funzioni di aumentazione
def rotate_image(image, angle):
    return image.rotate(angle)

def crop_image(image, crop_area):
    return image.crop(crop_area)

def zoom_image(image, zoom_factor):
    width, height = image.size
    x_center, y_center = width / 2, height / 2
    new_width, new_height = width / zoom_factor, height / zoom_factor
    left = x_center - new_width / 2
    top = y_center - new_height / 2
    right = x_center + new_width / 2
    bottom = y_center + new_height / 2
    return image.crop((left, top, right, bottom)).resize((width, height), Image.LANCZOS)

def shift_image(image, dx, dy):
    width, height = image.size
    return Image.fromarray(np.roll(np.roll(np.array(image), dx, axis=1), dy, axis=0), 'RGB')

def flip_image_horizontal(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def flip_image_vertical(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)

def random_rotation(image, max_angle):
    angle = random.uniform(-max_angle, max_angle)
    return rotate_image(image, angle)

def random_crop(image, crop_size):
    width, height = image.size
    left = random.randint(0, width - crop_size[0])
    top = random.randint(0, height - crop_size[1])
    right = left + crop_size[0]
    bottom = top + crop_size[1]
    return crop_image(image, (left, top, right, bottom))

def random_zoom(image, min_zoom, max_zoom):
    zoom_factor = random.uniform(min_zoom, max_zoom)
    return zoom_image(image, zoom_factor)

def random_shift(image, max_dx, max_dy):
    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)
    return shift_image(image, dx, dy)

def shear_image(image, shear_factor):
    width, height = image.size
    xshift = abs(shear_factor) * width
    new_width = width + int(round(xshift))
    return image.transform((new_width, height), Image.AFFINE,
                           (1, shear_factor, -xshift if shear_factor > 0 else 0, 0, 1, 0), Image.BICUBIC)

def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def adjust_saturation(image, factor):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)

def adjust_hue(image, factor):
    image = np.array(image.convert('HSV'))
    image[..., 0] = (image[..., 0].astype(int) + factor) % 256
    return Image.fromarray(image, 'HSV').convert('RGB')

def add_noise(image, noise_level):
    image = np.array(image)
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image, 'RGB')

def blur_image(image, radius):
    return image.filter(ImageFilter.GaussianBlur(radius))

def sharpen_image(image, factor):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)

def grayscale_image(image):
    return ImageOps.grayscale(image).convert('RGB')

def cutout_image(image, mask_size, mask_value=0):
    image = np.array(image)
    height, width = image.shape[:2]
    y = np.random.randint(height)
    x = np.random.randint(width)
    y1 = np.clip(y - mask_size // 2, 0, height)
    y2 = np.clip(y + mask_size // 2, 0, height)
    x1 = np.clip(x - mask_size // 2, 0, width)
    x2 = np.clip(x + mask_size // 2, 0, width)
    image[y1:y2, x1:x2] = mask_value
    return Image.fromarray(image, 'RGB')