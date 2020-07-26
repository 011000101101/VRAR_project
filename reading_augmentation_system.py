import segmentation.Segmentation as segmentation_subsystem
import classifier_subsystem.cnn_with_tfrecords as classifier_subsystem
import translator_subsystem.lut_translator as translator_subsystem
import utils.preprocess as pp
import utils.classify_util as classify_utils
import utils.image_augmenting as augment_utils
import GUI.detect_kanji as gui_module
from utils.params import *

import numpy as np
import cv2
import itertools
import pickle
import os


def show_rois(image_in: np.ndarray, rois_list_in: list):
    for column in rois_list_in:
        for roi in column:
            _, (x, y, w, h) = roi
            cv2.rectangle(image_in, (x, y), (x + w, y + h), (0, 255, 0))
    return image_in


def compute_rois(image_in: np.ndarray) -> (np.ndarray, list):
    return segmentation_subsystem.retrieve_current_frame(image_in)


def filter_rois_list(rois_list_in: list, roi_size: int) -> list:
    # filter small and large rois  # TODO
    rois_list_out = [pp.filter_roi_list(rois, roi_size) for rois in rois_list_in]
    rois_list_out = [rois for rois in rois_list_out if len(rois) > 0]
    return rois_list_out


def infer_readings(labels_list_in: list) -> list:
    return [
        translator_subsystem.translate_and_mask_line("".join([classify_utils.tensor_to_kanji(label) for label in line]))
        for
        line
        in
        labels_list_in
    ]


def fake_infer_readings(labels_list_in: list) -> list:
    # print("".join([classify_utils.tensor_to_kanji(label) for label in labels_list_in[0]]))
    return [
        [(
            [(1, kanji, kanji) for kanji in [classify_utils.tensor_to_kanji(label) for label in line]],
            len(line),
            len(line)
        )]
        for
        line
        in
        labels_list_in
    ]


def augment_samples(rois_list_in, readings_list_in) -> list:
    augmented_samples_out = [
        augment_utils.augment_samples(sample_line, reading_line)
        for
        sample_line, reading_line
        in
        zip(rois_list_in, readings_list_in)
    ]

    # flatten list of augmented samples and (x, y, w, h)
    augmented_samples_out = list(itertools.chain(*augmented_samples_out))
    augmented_samples_out = list(itertools.chain(*augmented_samples_out))

    return augmented_samples_out


def show_image_cv2(image: np.ndarray):
    cv2.imshow("augmented_with_readings", image)
    cv2.waitKey()
    cv2.destroyAllWindows()


class ReadingAugmentationSystem:

    def __init__(self):
        self.classifier = classifier_subsystem.load_model()
        self.gui = gui_module.detect_kanji(self.process_one_frame)

    def classify_image_samples(self, rois_list_in: list) -> list:
        return [
            self.classifier.predict(
                np.asarray(rois).reshape((len(rois), ) + rois[0].shape + (1, ))
            )
            for
            rois
            in
            rois_list_in
        ]

    def process_one_frame(self, frame: np.ndarray, roi_size: int):

        # compute rois
        current_frame, rois_list = compute_rois(frame)

        augmented_image_rois = show_rois(np.copy(frame), rois_list)  # TODO remove
        #
        # filter small and large rois  # TODO
        rois_list = filter_rois_list(rois_list, roi_size)
        #
        augmented_image_rois_filtered = show_rois(np.copy(frame), rois_list)  # TODO remove

        # make image samples square
        processed_rois_list = [pp.preprocess_roi_list(rois) for rois in rois_list]

        # resize image samples to common sample size for classifier
        resized_rois_list = [pp.resize_roi_list(rois) for rois in rois_list]

        # classify image samples
        labels = self.classify_image_samples(resized_rois_list)

        # # debug info
        lines = [
                "".join([classify_utils.tensor_to_kanji(label) for label in line])
                for
                line
                in
                labels
            ]
        print(lines)
        # string_labels = [
        #     [classify_utils.tensor_to_kanji(label) for label in line]
        #     for
        #     line
        #     in
        #     labels
        # ]
        # tagged_samples = [
        #     (sample, kanji)
        #     for
        #     sample, kanji
        #     in
        #     zip(
        #         [item for sublist in resized_rois_list for item in sublist],
        #         [item for sublist in string_labels for item in sublist]
        #     )
        # ]
        # with open(os.path.join(ROOT_DIR, "debug/classified_samples.pkl"), 'wb') as f:
        #     pickle.dump(tagged_samples, f, pickle.HIGHEST_PROTOCOL)

        # infer readings
        readings = infer_readings(labels)
        # print(readings)

        # transform detected kanji into "readings" format to augment the detected kanji themselves instead of
        # their readings
        kanji_fake_readings = fake_infer_readings(labels)

        # augment readings into image samples  # TODO
        augmented_samples_fake = augment_samples(processed_rois_list, kanji_fake_readings)  # TODO
        # TODO
        # recombine with image  # TODO
        augmented_image_fake = augment_utils.recombine(np.copy(frame), augmented_samples_fake)  # TODO
        # TODO
        # cv2.imshow("augmented_with_labels", augmented_image_fake)  # TODO
        # cv2.waitKey()  # TODO
        # cv2.destroyAllWindows()  # TODO

        # augment readings into image samples
        augmented_samples = augment_samples(processed_rois_list, readings)

        # recombine with image
        augmented_image = augment_utils.recombine(frame, augmented_samples)

        # show_image_cv2(augmented_image)
        return augmented_image_rois_filtered


if __name__ == "__main__":

    system = ReadingAugmentationSystem()

    # # get frame
    # current_frame_raw = cv2.imread("./media/Manga_raw.jpg")
    #
    # while True:
    #
    #     system.process_one_frame(np.copy(current_frame_raw))

