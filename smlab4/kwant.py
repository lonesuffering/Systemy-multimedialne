import cv2
import numpy as np
import matplotlib.pyplot as plt

def color_fit(pixel, palette):
    pixel = np.array(pixel)
    distances = np.linalg.norm(palette - pixel, axis=1)
    return palette[np.argmin(distances)]

def quantize_image(image, palette):
    quantized_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            quantized_image[i, j] = color_fit(image[i, j], palette)

    return quantized_image

def is_grayscale(image):
    if len(image.shape) == 2: 
        return True
    elif len(image.shape) == 3:
        return np.all(image[:, :, 0] == image[:, :, 1]) and np.all(image[:, :, 1] == image[:, :, 2])
    return False


def process_and_display(image_path):
    image = cv2.imread(image_path).astype(np.float32) / 255.0
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    if is_grayscale(image):
        palette_1bit = np.linspace(0, 1, 2).reshape(2, 1)
        palette_2bit = np.linspace(0, 1, 4).reshape(4, 1)
        palette_4bit = np.linspace(0, 1, 16).reshape(16, 1)
        quantized_1bit = quantize_image(image, palette_1bit)
        quantized_2bit = quantize_image(image, palette_2bit)
        quantized_4bit = quantize_image(image, palette_4bit)
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        titles = ['Original', 'Pallet 1-bit', 'Pallet 2-bit', 'Pallet 4-bit']
        images = [image, quantized_1bit, quantized_2bit, quantized_4bit]

    else:  
        pallet8 = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ])

        pallet16 = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.5],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [1.0, 0.0, 0.0],
            [0.75, 0.75, 0.75],
            [0.0, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ])

        quantized_8 = quantize_image(image_rgb, pallet8)
        quantized_16 = quantize_image(image_rgb, pallet16)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ['Original', 'Pallet 8 Colors', 'Pallet 16 Colors']
        images = [image_rgb, quantized_8, quantized_16]
        
    for ax, img, title in zip(axes, images, titles):
        if len(img.shape) == 2: 
            ax.imshow(np.clip(img, 0, 1), cmap='gray')
        else: 
            ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title)
        ax.axis('off')

    plt.show()
    
process_and_display('IMG_GS/GS_0001.tif')  # для серого изображения
process_and_display('IMG_GS/GS_0002.png')
process_and_display('IMG_GS/GS_0003.png')

process_and_display('IMG_SMALL/SMALL_0001.tif')  # для цветного изображения
process_and_display('IMG_SMALL/SMALL_0009.jpg')
process_and_display('IMG_SMALL/SMALL_0004.jpg')
process_and_display('IMG_SMALL/SMALL_0006.jpg')
process_and_display('IMG_SMALL/SMALL_0007.jpg')