import segmentation.Segmentation as segmentation_subsystem
import classifier_subsystem.cnn as classifier_subsystem
import translator_subsystem.translator_subsystem as translator_subsystem
import utils.preprocess as pp
import utils.classify_util as classify_utils
import utils.image_augmenting as augment_utils

import numpy as np
import cv2
import itertools

if __name__ == "__main__":

    classifier = classifier_subsystem.load_model()

    while True:
        current_frame_raw = cv2.imread("./media/Manga_raw.jpg")
        # get frame
        # current_frame_raw = segmentation_subsystem.get_camera_image()
        # compute rois
        current_frame, rois_list = segmentation_subsystem.retrieve_current_frame(current_frame_raw)  # , debugLevel=1)
        cv2.imshow("augmented", current_frame)
        cv2.waitKey()
        cv2.destroyAllWindows()
        # filter small and large rois  # TODO
        rois_list = [pp.filter_roi_list(rois) for rois in rois_list]
        rois_list = [rois for rois in rois_list if len(rois) > 0]
        # make image samples square and resample to common size
        processed_rois_list = [pp.preprocess_roi_list(rois) for rois in rois_list]
        # resize image samples to common sample size for classifier
        resized_rois_list = [pp.resize_roi_list(rois) for rois in rois_list]
        # classify image samples
        labels = [
            classifier.predict(
                np.asarray(rois).reshape((len(rois), ) + rois[0].shape + (1, ))
            )
            for
            rois
            in
            resized_rois_list
        ]
        # print(labels)
        # infer readings
        # print("".join([classify_utils.tensor_to_kanji(label) for label in labels[0]]))
        readings = [
            translator_subsystem.translate_and_mask_line("".join([classify_utils.tensor_to_kanji(label) for label in line]))
            for
            line
            in
            labels
        ]
        # readings = [
        #     [(
        #         [(1, kanji, kanji) for kanji in [classify_utils.tensor_to_kanji(label) for label in line]],
        #         len(line),
        #         len(line)
        #     )]
        #     for
        #     line
        #     in
        #     labels
        # ]
        print(len(list(itertools.chain(*[asdf[0][0] for asdf in readings]))))
        print(readings)
        print(len(readings))
        # augment readings into image samples
        augmented_samples = [
            augment_utils.augment_samples(sample_line, reading_line)
            for
            sample_line, reading_line
            in
            zip(processed_rois_list, readings)
        ]
        print(len(augmented_samples))
        # flatten list of augmented samples and (x, y, w, h)
        augmented_samples = list(itertools.chain(*augmented_samples))
        augmented_samples = list(itertools.chain(*augmented_samples))
        # print(len(augmented_samples))
        # for img in augmented_samples:
        #     cv2.imshow("sample", img[0])
        #     cv2.waitKey()
        #     cv2.destroyAllWindows()
        # recombine with image
        # print(augmented_samples)
        augmented_image = augment_utils.recombine(current_frame_raw, augmented_samples)
        #call gui  # TODO
        cv2.imshow("augmented", augmented_image)
        cv2.waitKey()
        cv2.destroyAllWindows()

