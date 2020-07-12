import numpy as np
import os
import cv2
import pickle
from utils.image_augmenting import add_noise_greyscale
import os
from utils.params import *


def augment(image_samples):
    """
    augments each of the given images by addying noise, blur and both
    :param image_samples:  image samples as dict indexed by the kanji the image depicts, with a list of
                            (font_name, image) as value
    :return: a list containing the original as well as the augmented image samples with the kanji they depict
    """
    samples = []
    clean_samples = []

    i= 0
    print(len(image_samples.keys()))
    for kanji in image_samples.keys():
        print(i)
        i += 1
        for image in image_samples[kanji]:
            img_gauss = add_noise_greyscale(image[1], "gauss")
            img_s_p = add_noise_greyscale(image[1], "s_p")
            img_speckle = add_noise_greyscale(image[1], "speckle")
            img_all = add_noise_greyscale(add_noise_greyscale(img_speckle, "gauss"), "s_p")
            img_all_blur_median = cv2.medianBlur(image[1].astype('uint8'), 3)
            img_all_blur_gauss = cv2.GaussianBlur(image[1].astype('uint8'), (0, 0), 0.3)
            img_all_blur_bilateral = cv2.bilateralFilter(image[1].astype('uint8'), 3, 50, 50)
            upper_row = np.hstack((image[1], img_gauss, img_s_p, img_speckle))
            lower_row = np.hstack((img_all, img_all_blur_median, img_all_blur_gauss, img_all_blur_bilateral))
            total = np.vstack((upper_row, lower_row))
            # cv2.imshow("asdf", total);cv2.waitKey();cv2.destroyAllWindows()

            samples.append((image[1], kanji))
            samples.append((img_gauss, kanji))
            samples.append((img_s_p, kanji))
            samples.append((img_speckle, kanji))
            samples.append((img_all, kanji))
            samples.append((img_all_blur_median, kanji))
            samples.append((img_all_blur_gauss, kanji))
            samples.append((img_all_blur_bilateral, kanji))

            clean_samples.append((image[1], kanji))

    return samples, clean_samples


if __name__ == "__main__":

    # load clean image samples from disk
    with open(os.path.join(ROOT_DIR, "bin_blobs/kanji_image_samples.pkl"), 'rb') as f:
        image_samples = pickle.load(f)

    # augment
    samples, clean_samples = augment(image_samples)

    # save augmented image samples to disk
    with open(os.path.join(ROOT_DIR, "bin_blobs/kanji_image_samples_augmentes.pkl"), 'wb') as f:
        pickle.dump(samples, f, pickle.HIGHEST_PROTOCOL)

    # save augmented image samples to disk
    with open(os.path.join(ROOT_DIR, "bin_blobs/kanji_image_samples_clean.pkl"), 'wb') as f:
        pickle.dump(clean_samples, f, pickle.HIGHEST_PROTOCOL)
