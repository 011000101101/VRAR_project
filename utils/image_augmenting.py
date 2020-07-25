import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
import os
from utils.params import *


def add_noise_greyscale(image: np.ndarray, noise_typ: str):
    """
    source: https://stackoverflow.com/a/30609854
    adds variable noise to an image
    :param noise_typ: One of the following strings, selecting the type of noise to add:
                        'gauss'     Gaussian-distributed additive noise.
                        'poisson'   Poisson-distributed noise generated from the data.
                        's_p'       Replaces random pixels with 0 or 1.
                        'speckle'   Multiplicative noise using out = image + n*image,where
                                    n is uniform noise with specified mean & variance.
    :param image: Input image data. Will be converted to float.
    :return:
    """
    if noise_typ == "gauss":
        shape = image.shape
        mean = 0.03
        var = 0.3
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, shape)
        gauss = gauss.reshape(shape)
        noisy = image + gauss
        return noisy.clip(0, 255).astype("uint8")
    elif noise_typ == "s_p":
        s_vs_p = 0.5
        amount = 0.02  # 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[tuple(coords)] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[tuple(coords)] = 0
        return out.clip(0, 255).astype("uint8")
    elif noise_typ == "speckle":
        w, h = image.shape
        gauss = np.random.randn(w, h)
        gauss = gauss.reshape((w, h))
        gauss = gauss / 5
        noisy = image + image * gauss
        return noisy.clip(0, 255).astype("uint8")


def blur_custom(img: np.ndarray) -> np.ndarray:
    kernel = np.asarray([
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625]
    ])
    return cv2.filter2D(img, -1, kernel)


def blur_avg(img: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.float32) / 9
    return cv2.filter2D(img, -1, kernel)


def blur_gauss(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img.astype('uint8'), (3, 3), 0.0).astype('uint8')


def apply_blur_3_times(img: np.ndarray, blur_func: callable) -> list:
    tmp = []
    img_tmp = img
    for _ in range(3):
        img_tmp = blur_func(img_tmp)
        tmp.append(img_tmp)
    return tmp


def augment_samples(samples, readings):
    current_sample_index = 0
    augmented_segments = []
    for segment in readings:
        words, segment_len, n_leading_kanji = segment
        if words == [] or n_leading_kanji == 0:  # skip lines without readings or without kanji
            current_sample_index += segment_len
            continue
        augmented_segments.append(
            augment_segment(
                samples[current_sample_index:current_sample_index+segment_len],
                [
                    (reb, n_symbols, keb)
                    for
                    n_symbols, keb, reb
                    in
                    words
                    if
                    n_symbols != 0
                ],
                n_leading_kanji,
            )
        )
        current_sample_index += segment_len
    return augmented_segments


def augment_segment(samples, words, n_leading_kanji):
    current_sample_index = 0
    augmented_words = []
    for reb, n_symbols, keb in words:
        augmented_words.append(
            augment_word(
                samples[current_sample_index:current_sample_index+min(n_symbols, n_leading_kanji)],
                reb[:len(reb)-(len(keb)-min(n_symbols, n_leading_kanji))]
            )
        )
        current_sample_index += min(n_symbols, n_leading_kanji)
        n_leading_kanji -= n_symbols
        if n_leading_kanji <= 0:
            break
    return augmented_words


def augment_word(samples: list, reb: str):
    """
    fuse the images of a word and augment it with its reading
    :param samples: list of images of the single kanji
    :param reb: reading of this word
    :return:
    """
    # create canvas for fused images
    # coordinates of first image
    _, (x, y, _, _) = samples[0]
    # coordinates and size of last image
    _, (x_tmp, y_tmp, w_tmp, h_tmp) = samples[-1]
    # compute width and height of new image
    w = (x_tmp + w_tmp) - x
    h = (y_tmp + h_tmp) - y
    # create white image
    new_sample = np.full((h, w), 255, dtype="uint8")

    # place each kanji at half scale into the left half of its original bounding box
    for sample in samples:
        # unpack kanji image
        image, (x_tmp, y_tmp, w_tmp, h_tmp) = sample
        # get relative coordinates of bounding box
        x_tmp = 0  # ignore slight left/right shifts (might move kanji by a few pixels)
        y_tmp = y_tmp - y
        # compute new width and height
        new_w_tmp = int(w_tmp/2)
        new_h_tmp = int(h_tmp/2)
        # rescale and insert
        new_sample[y_tmp:y_tmp+new_h_tmp, x_tmp:x_tmp+new_w_tmp] = cv2.resize(image, (new_h_tmp, new_w_tmp))

    # compute font size (maximum possible given available space and assuming all kanji are square
    font_size_px = int(min(w/2, h/len(reb)))
    font_size_pt = int(font_size_px * 1)  # 0.75)  # convert to pt
    # prepare image for text rendering
    img_pil = Image.fromarray(new_sample[:, int(w/2):])
    draw = ImageDraw.Draw(img_pil)
    augmentation_font = ImageFont.truetype(os.path.join(ROOT_DIR, "resources/fonts/Xano-Mincho/XANO-mincho-U32.ttf"), font_size_pt)
    # compute initial vertical position
    y_tmp = (h-font_size_px*len(reb)) // 2
    # render each kanji by putting text onto the image
    for char in reb:
        draw.text((0, y_tmp), char, font=augmentation_font, fill=0)  # TODO center in segment
        # increment vertical position
        y_tmp += font_size_px
    # convert image back to numpy array and merge into augmented image
    new_sample[:, int(w/2):] = np.array(img_pil)
    return new_sample, (x, y, w, h)


def recombine(current_frame_raw, augmented_samples):
    for sample in augmented_samples:
        image, (x, y, w, h) = sample
        try:
            current_frame_raw[y:y+h, x:x+w] = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        except ValueError:  # trying to paste outside of image area...
            if y + h > current_frame_raw.shape[0]:
                h = current_frame_raw.shape[0] - y
                image = image[:h, :]
            if x + w > current_frame_raw.shape[1]:
                w = current_frame_raw.shape[1] - x
                image = image[:, :w]
            if x < 0:
                w = w + x
                image = image[:, -x:]
                x = 0
            if y < 0:
                h = h + y
                image = image[-y:, :]
                y = 0
            current_frame_raw[y:y+h, x:x+w] = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return current_frame_raw
