import cv2
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter


def start():
    pics = {'Clusters', 'plane1', 'plane2'}
    for pic in pics:
        refactor_pic(pic)


def refactor_pic(name):
    img = loading_img("data/input/%s.jpg" % name)
    # final_img = delete_noise_midpoint(img)
    # final_img = delete_noise_kernel(final_img)
    final_img = delete_noise_contraharmonic_mean(img)
    final_img = unsharp_mask(final_img)
    save_img(name, final_img)


def delete_noise_midpoint(img):
    max_f = maximum_filter(img, size=3)
    min_f = minimum_filter(img, size=3)
    return (max_f + min_f) / 2


def delete_noise_contraharmonic_mean(img):
    size = (3, 3)
    q = 0.5
    num = np.power(img, q + 1)
    denom = np.power(img, q)
    kernel = np.full(size, 1.0)
    result1 = cv2.filter2D(num, -1, kernel)
    result2 = cv2.filter2D(denom, -1, kernel)
    return cv2.divide(result1, result2)


def unsharp_mask(img, kernel_size=(3, 3), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = float(amount + 1) * img - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(img - blurred) < threshold
        np.copyto(sharpened, img, where=low_contrast_mask)
    return sharpened


def delete_noise_kernel(img):
    kernel_sharpening = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    return cv2.filter2D(img, -1, kernel_sharpening)


def show_img(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)


def save_img(name, img):
    cv2.imwrite("data/output/%s.jpg" % name, img)


def loading_img(path):
    img = cv2.imread(path)
    # ret, thresh = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
    return img


if __name__ == '__main__':
    start()
