import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

def imgToUInt8(img):
    if not np.issubdtype(img.dtype, np.floating):
        return img
    return (img * 255).astype('uint8')

def imgToFloat(img):
    if not np.issubdtype(img.dtype, np.unsignedinteger):
        return img
    return img / 255.0

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

df = pd.DataFrame(data={
    'Filename': ['B01.png', 'B02.jpg'],
    'Grayscale': [False, False],
    'Fragments': [[[20, 20, 220, 220], [100, 100, 300, 300]], [[50, 50, 250, 250]]]
})

with PdfPages('output.pdf') as pdf:
    for index, row in df.iterrows():
        img = plt.imread(row['Filename'])
        fragments = row['Fragments']

        for i, fragment_coords in enumerate(fragments):
            w1, k1, w2, k2 = fragment_coords
            fragment = img[w1:w2, k1:k2].copy()

            visualize_image(fragment)

            pdf.savefig()

            plt.close()