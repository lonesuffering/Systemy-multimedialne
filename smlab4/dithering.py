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

def dithering_random(image):
    random_matrix = np.random.rand(image.shape[0], image.shape[1], 1)
    random_matrix = np.repeat(random_matrix, image.shape[2], axis=2) 
    return (image >= random_matrix).astype(np.float32)

def dithering_ordered(image, palette, size=4, r=0.6):
    matrix = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) 
    height, width = image.shape[:2]
    output = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            pixel = image[i, j]
            matrix_value = matrix[i % size, j % size] / 16.0
            adjusted_pixel = pixel + r * matrix_value
            output[i, j] = color_fit(adjusted_pixel, palette)

    return output

def dithering_floyd_steinberg(image, palette):
    out = image.copy()

    if len(image.shape) == 2: 
        h, w = image.shape
        for y in range(h):
            for x in range(w):
                old = out[y, x]
                new = color_fit(np.array([old]), palette)
                out[y, x] = new[0]  
                err = old - new[0]
                if x + 1 < w: out[y, x + 1] += err * 7 / 16
                if x > 0 and y + 1 < h: out[y + 1, x - 1] += err * 3 / 16
                if y + 1 < h: out[y + 1, x] += err * 5 / 16
                if x + 1 < w and y + 1 < h: out[y + 1, x + 1] += err * 1 / 16
    elif len(image.shape) == 3: 
        h, w, c = image.shape  # h = height, w = width, c = channels (RGB)
        for y in range(h):
            for x in range(w):
                for channel in range(c):  
                    old = out[y, x, channel]
                    new = color_fit(np.array([old]), palette)
                    out[y, x, channel] = new[0] 
                    err = old - new[0]
                    if x + 1 < w: out[y, x + 1, channel] += err * 7 / 16
                    if x > 0 and y + 1 < h: out[y + 1, x - 1, channel] += err * 3 / 16
                    if y + 1 < h: out[y + 1, x, channel] += err * 5 / 16
                    if x + 1 < w and y + 1 < h: out[y + 1, x + 1, channel] += err * 1 / 16

    return np.clip(out, 0, 1)

def process_and_display(image_path):
    image = cv2.imread(image_path).astype(np.float32) / 255.0
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if is_grayscale(image):
        # Палитры для 1, 2 и 4 бит
        palette_1bit = np.linspace(0, 1, 2).reshape(2, 1)
        palette_2bit = np.linspace(0, 1, 4).reshape(4, 1)
        palette_4bit = np.linspace(0, 1, 16).reshape(16, 1)

        quantized_1bit = quantize_image(image, palette_1bit)
        quantized_2bit = quantize_image(image, palette_2bit)
        quantized_4bit = quantize_image(image, palette_4bit)

        dithering_1bit = dithering_random(image)
        dithering_ordered_1bit = dithering_ordered(image, palette_1bit)
        dithering_floyd_1bit = dithering_floyd_steinberg(image, palette_1bit)

        dithering_ordered_2bit = dithering_ordered(image, palette_2bit)
        dithering_floyd_2bit = dithering_floyd_steinberg(image, palette_2bit)

        dithering_ordered_4bit = dithering_ordered(image, palette_4bit)
        dithering_floyd_4bit = dithering_floyd_steinberg(image, palette_4bit)

        # Плот для 1 бита
        fig1, axes1 = plt.subplots(1, 5, figsize=(20, 5))
        titles_1bit = ['Original', 'Kwantyzacja', 'Dithering Zorganizawany', 'Dithering Losowy', 'Dithering Floyda-Steinberga']
        images_1bit = [image, quantized_1bit, dithering_ordered_1bit, dithering_1bit, dithering_floyd_1bit]

        for ax, img, title in zip(axes1, images_1bit, titles_1bit):
            ax.imshow(np.clip(img, 0, 1), cmap='gray')
            ax.set_title(title)
            ax.axis('off')

        # Плот для 2 бит
        fig2, axes2 = plt.subplots(1, 5, figsize=(20, 5))
        titles_2bit = ['Original', 'Kwantyzacja', 'Dithering Zorganizawany', 'Dithering Floyda-Steinberga']
        images_2bit = [image, quantized_2bit, dithering_ordered_2bit, dithering_floyd_2bit]

        for ax, img, title in zip(axes2, images_2bit, titles_2bit):
            ax.imshow(np.clip(img, 0, 1), cmap='gray')
            ax.set_title(title)
            ax.axis('off')

        # Плот для 4 бит
        fig3, axes3 = plt.subplots(1, 5, figsize=(20, 5))
        titles_4bit = ['Original', 'Kwantyzacja', 'Dithering Zorganizawany', 'Dithering Floyda-Steinberga']
        images_4bit = [image, quantized_4bit, dithering_ordered_4bit, dithering_floyd_4bit]

        for ax, img, title in zip(axes3, images_4bit, titles_4bit):
            ax.imshow(np.clip(img, 0, 1), cmap='gray')
            ax.set_title(title)
            ax.axis('off')

    else:
        palette8 = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ])

        palette16 = np.array([
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

        quantized_8 = quantize_image(image_rgb, palette8)
        quantized_16 = quantize_image(image_rgb, palette16)

        dithering_ordered_8 = dithering_ordered(image_rgb, palette8)
        dithering_floyd_8 = dithering_floyd_steinberg(image_rgb, palette8)

        dithering_ordered_16 = dithering_ordered(image_rgb, palette16)
        dithering_floyd_16 = dithering_floyd_steinberg(image_rgb, palette16)

        # Плот для 8 и 16 бит
        fig4, axes4 = plt.subplots(1, 5, figsize=(20, 5))

        titles_8 = ['Original', 'Kwantyzacja', 'Dithering Zorganizawany', 'Dithering Floyda-Steinberga']
        images_8 = [image_rgb, quantized_8, dithering_ordered_8, dithering_floyd_8]

        for ax, img, title in zip(axes4, images_8, titles_8):
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(title)
            ax.axis('off')

        fig5, axes5 = plt.subplots(1, 5, figsize=(20, 5))
        titles_16 = ['Original', 'Kwantyzacja', 'Dithering Zorganizawany', 'Dithering Floyda-Steinberga']
        images_16 = [image_rgb, quantized_16, dithering_ordered_16, dithering_floyd_16]

        for ax, img, title in zip(axes5, images_16, titles_16):
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(title)
            ax.axis('off')

    plt.show()


# Пример использования
process_and_display('IMG_GS/GS_0001.tif')  # для серого изображения
process_and_display('IMG_GS/GS_0002.png')
process_and_display('IMG_GS/GS_0003.png')

process_and_display('IMG_SMALL/SMALL_0001.tif')  # для цветного изображения
process_and_display('IMG_SMALL/SMALL_0009.jpg')
process_and_display('IMG_SMALL/SMALL_0004.jpg')
process_and_display('IMG_SMALL/SMALL_0006.jpg')
process_and_display('IMG_SMALL/SMALL_0007.jpg')
