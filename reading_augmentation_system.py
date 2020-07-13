import segmentation.Segmentation as segmentation_subsystem
import classifier_subsystem.cnn as classifier_subsystem
import translator_subsystem.translator_subsystem as translator_subsystem
import utils.preprocess as pp
import utils.classify_util as classify_utils
import utils.image_augmenting as augment_utils

import numpy as np
import cv2
import itertools


def show_rois(image_in: np.ndarray, rois_list_in: list):
    for column in rois_list_in:
        for roi in column:
            _, (x, y, w, h) = roi
            cv2.rectangle(image_in, (x, y), (x + w, y + h), (0, 255, 0))
    cv2.imshow("rois", image_in)
    cv2.waitKey()
    cv2.destroyAllWindows()


def compute_rois(image_in: np.ndarray) -> (np.ndarray, list):
    return segmentation_subsystem.retrieve_current_frame(image_in)


def filter_rois_list(rois_list_in: list) -> list:
    # filter small and large rois  # TODO
    rois_list_out = [pp.filter_roi_list(rois) for rois in rois_list_in]
    rois_list_out = [rois for rois in rois_list_out if len(rois) > 0]
    return rois_list_out


def classify_image_samples(rois_list_in: list) -> list:
    return [
        classifier.predict(
            np.asarray(rois).reshape((len(rois), ) + rois[0].shape + (1, ))
        )
        for
        rois
        in
        resized_rois_list
    ]


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


if __name__ == "__main__":

    classifier = classifier_subsystem.load_model()

    while True:

        # get frame
        current_frame_raw = cv2.imread("./media/Manga_raw.jpg")
        # current_frame_raw = segmentation_subsystem.get_camera_image()

        # compute rois
        current_frame, rois_list = compute_rois(current_frame_raw)

        show_rois(np.copy(current_frame_raw), rois_list)  # TODO remove

        # filter small and large rois  # TODO
        rois_list = filter_rois_list(rois_list)

        show_rois(np.copy(current_frame_raw), rois_list)  # TODO remove

        # make image samples square
        processed_rois_list = [pp.preprocess_roi_list(rois) for rois in rois_list]

        # resize image samples to common sample size for classifier
        resized_rois_list = [pp.resize_roi_list(rois) for rois in rois_list]

        # classify image samples
        labels = classify_image_samples(resized_rois_list)

        # infer readings
        readings = infer_readings(labels)

        # transform detected kanji into "readings" format to augment the detected kanji themselves instead of
        # their readings
        kanji_fake_readings = fake_infer_readings(labels)

        # augment readings into image samples  # TODO
        augmented_samples_fake = augment_samples(processed_rois_list, kanji_fake_readings)  # TODO
        # TODO
        # recombine with image  # TODO
        augmented_image_fake = augment_utils.recombine(np.copy(current_frame_raw), augmented_samples_fake)  # TODO
        # TODO
        cv2.imshow("augmented_with_labels", augmented_image_fake)  # TODO
        cv2.waitKey()  # TODO
        cv2.destroyAllWindows()  # TODO

        # augment readings into image samples
        augmented_samples = augment_samples(processed_rois_list, readings)

        # recombine with image
        augmented_image = augment_utils.recombine(current_frame_raw, augmented_samples)

        #call gui  # TODO
        cv2.imshow("augmented_with_readings", augmented_image)
        cv2.waitKey()
        cv2.destroyAllWindows()

