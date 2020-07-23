import numpy as np
import os
import cv2
import pickle
from utils.image_augmenting import *
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

            samples_tmp = []

            # create noisy image
            img_all = add_noise_greyscale(add_noise_greyscale(add_noise_greyscale(image[1], "speckle"), "gauss"), "s_p")

            samples_tmp.append(image[0])
            samples_tmp.append(img_all)

            # create various blurs in 3 intensities on the clean and noisy image
            # img_all_blur_median = cv2.medianBlur(img_all.astype('uint8'), 3)  # bad, loses details

            samples_tmp += apply_blur_3_times(image[1], blur_custom)
            samples_tmp += apply_blur_3_times(img_all, blur_custom)

            samples_tmp += apply_blur_3_times(image[1], blur_avg)
            samples_tmp += apply_blur_3_times(img_all, blur_avg)

            samples_tmp += apply_blur_3_times(image[1], blur_gauss)
            samples_tmp += apply_blur_3_times(img_all, blur_gauss)

            # img_all_blur_bilateral = cv2.bilateralFilter(img_all.astype('uint8'), 3, 50, 50)  # bad, doesn't do much

            source_row = np.hstack([image[1], image[1], image[1], img_all, img_all, img_all])
            upper_row = np.hstack(samples_tmp[2:8])
            mid_row = np.hstack(samples_tmp[8:14])
            lower_row = np.hstack(samples_tmp[14:20])
            total = np.vstack((source_row, upper_row, mid_row, lower_row))
            cv2.imshow("asdf", total);cv2.waitKey();cv2.destroyAllWindows()

            samples += [(img, kanji) for img in samples_tmp]

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
