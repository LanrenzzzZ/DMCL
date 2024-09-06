import numpy as np
from skimage.util.shape import view_as_windows

def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0,:,:, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs

def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    return image

def random_crop_2(imgs, output_size):

    n1 = imgs.shape[0]  # 16
    n2 = imgs.shape[1]  # 64
    n = n1 * n2
    imgs = imgs.reshape(n, *imgs.shape[2:])  # (1024, 3, 100, 140)
    img_size = imgs.shape[-1]  # 140
    crop_max = img_size - output_size  # 140-84=56
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0, :, :, 0]
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs.reshape(n1, n2, *cropped_imgs.shape[1:])