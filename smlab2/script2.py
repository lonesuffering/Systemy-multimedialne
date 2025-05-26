import numpy as np
import matplotlib.pyplot as plt

def visualize_image(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    Y1 = R
    Y2 = 0.2126 * R + 0.7152 * G + 0.0722 * B

    img_R = img.copy()
    img_G = img.copy()
    img_B = img.copy()
    img_R[:, :, 1:] = 0
    img_G[:, :, [0, 2]] = 0
    img_B[:, :, :2] = 0

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 3, 1)
    plt.title("O - Original")
    plt.imshow(img)

    plt.subplot(3, 3, 2)
    plt.title("Y1 - Grayscale (Simple)")
    plt.imshow(Y1, cmap='gray')

    plt.subplot(3, 3, 3)
    plt.title("Y2 - Grayscale (Weighted)")
    plt.imshow(Y2, cmap='gray')

    # Второй ряд: Каналы R, G, B
    plt.subplot(3, 3, 4)
    plt.title("R - Red Channel")
    plt.imshow(R, cmap='gray')

    plt.subplot(3, 3, 5)
    plt.title("G - Green Channel")
    plt.imshow(G, cmap='gray')

    plt.subplot(3, 3, 6)
    plt.title("B - Blue Channel")
    plt.imshow(B, cmap='gray')

    plt.subplot(3, 3, 7)
    plt.title("R Only")
    plt.imshow(img_R)

    plt.subplot(3, 3, 8)
    plt.title("G Only")
    plt.imshow(img_G)

    plt.subplot(3, 3, 9)
    plt.title("B Only")
    plt.imshow(img_B)

    plt.tight_layout()
    plt.show()

img = plt.imread('B01.png')

visualize_image(img)