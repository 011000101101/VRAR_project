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
        gauss = gauss / 3
        noisy = image + image * gauss
        return noisy.clip(0, 255).astype("uint8")


def augment_samples(samples, readings):
    current_sample_index = 0
    augmented_segments = []
    for segment in readings:
        words, segment_len, n_leading_kanji = segment
        print(words)
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
    _, (x, y, _, _) = samples[0]
    _, (x_tmp, y_tmp, w_tmp, h_tmp) = samples[-1]
    w = (x_tmp + w_tmp) - x
    h = (y_tmp + h_tmp) - y
    new_sample = np.full((h, w), 255, dtype="uint8")
    for sample in samples:
        image, (x_tmp, y_tmp, w_tmp, h_tmp) = sample
        x_tmp = x_tmp - x
        y_tmp = y_tmp - y
        new_w_tmp = int(w_tmp/2)
        new_h_tmp = int(h_tmp/2)
        new_sample[y_tmp:y_tmp+new_h_tmp, x_tmp:x_tmp+new_w_tmp] = cv2.resize(image, (new_h_tmp, new_w_tmp))
    font_size_px = int(min(w/2, h/len(reb)))
    font_size_pt = int(font_size_px * 1)  # 0.75)  # convert to pt
    # convert to correct format
    img_pil = Image.fromarray(new_sample[:, int(w/2):])
    # render single kanji by putting text onto the image
    draw = ImageDraw.Draw(img_pil)
    augmentation_font = ImageFont.truetype(os.path.join(ROOT_DIR, "resources/fonts/MODI_komorebi-gothic_2018_0501/komorebi-gothic.ttf"), font_size_pt)
    y_tmp = int((w-font_size_px*len(reb))/2)
    print("reb: " + reb)
    for char in reb:
        print(char)
        draw.text((0, y_tmp), char, font=augmentation_font, fill=0)  # TODO center in segment
        y_tmp += font_size_px
    # convert image back to numpy array and merge into augmented image
    new_sample[:, int(w/2):] = np.array(img_pil)
    return new_sample, (x, y, w, h)


def recombine(current_frame_raw, augmented_samples):
    for sample in augmented_samples:
        image, (x, y, w, h) = sample
        current_frame_raw[y:y+h, x:x+w] = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return current_frame_raw
