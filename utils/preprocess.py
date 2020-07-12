from utils.params import *

import numpy as np
import cv2


def square_sample(sample: np.ndarray) -> np.ndarray:
    w, h = sample.shape
    new_size = max(w, h)
    new_sample = np.full((new_size, new_size), 255, dtype=sample.dtype)
    new_sample[int((new_size-w)/2):int((new_size-w)/2+w), int((new_size-h)/2):int((new_size-h)/2+h)] = sample
    return new_sample


def unify_sample_size(sample: np.ndarray) -> np.ndarray:
    return cv2.resize(sample, (SAMPLE_IMAGE_SIZE, SAMPLE_IMAGE_SIZE))


def preprocess_roi_list(rois: list):

    split_rois = []
    for image, (x, y, w, h) in rois:
        vertical_stacking_factor = h/w
        while vertical_stacking_factor > 1.75:
            split_rois.append(
                (
                    image[y:y + w, x:x + w],
                    (x, y, w, w)
                )
            )
            vertical_stacking_factor = vertical_stacking_factor - 1
            y = y + w
            h = h - w
        split_rois.append(
            (
                image[y:y + h, x:x + w],
                (x, y, w, h)
            )
        )

    processed_rois = [
        (unify_sample_size(square_sample(image)), location)
        for
        image, location
        in
        split_rois
    ]

    return processed_rois


if __name__ == "__main__":

    sample = np.zeros((30, 15))
    cv2.imshow("original", sample)
    cv2.waitKey()
    cv2.destroyAllWindows()

    square_sample = square_sample(sample)
    cv2.imshow("square", square_sample)
    cv2.waitKey()
    cv2.destroyAllWindows()

    unified_image = unify_sample_size(square_sample)
    cv2.imshow("large", unified_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
