import segmentation.Segmentation as segmentation_subsystem
import classifier_subsystem.cnn as classifier_subsystem
import translator_subsystem.translator_subsystem as translator_subsystem
import utils.preprocess as pp

import numpy as np
import cv2

while True:
    # get frame and rois
    current_frame, rois_list = segmentation_subsystem.retrieve_current_frame()
    # make image samples square and resample to common size
    processed_rois_list = [pp.preprocess_roi_list(rois) for rois in rois_list]
    # classify image samples
    labels = [
        classifier_subsystem.classify([sample for sample, location in rois])
        for
        rois
        in
        processed_rois_list
    ]
    # infer readings
    readings = [
        translator_subsystem.translate_and_mask_sequence("".join(kanjis))
        for
        kanjis
        in
        labels
    ]
    # augment readings into image samples
    augmented_samples = [

    ]
    # recombine with image

    #call gui

