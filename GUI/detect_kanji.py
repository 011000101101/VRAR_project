import sys

import cv2
from PyQt5.QtCore import QTimer, Qt, QRect
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from PyQt5.uic import loadUi
import numpy as np

from utils.params import *
mode_dict = { 0: "segment", 1: "classify", 2: "augment"}

def move_ui_object(ui_object, height_offset, width_offset):
    ui_object.setGeometry(QRect(
        ui_object.x() + width_offset,
        ui_object.y() + height_offset,
        ui_object.width(),
        ui_object.height()
    ))


class detect_kanji(QDialog):
    def __init__(self, process_frame_func):
        self.app = QApplication(sys.argv)

        super(detect_kanji, self).__init__()
        loadUi(os.path.join(ROOT_DIR, 'GUI/detectkanji.ui'), self)

        self.roi_size = 25
        self.mode = 2
        self.timer = QTimer(self)
        self.label.setText(str(self.roi_size))
        self.label_2.setText("augment")
        self.process_frame = process_frame_func
        self.image = None
        self.processed_image = None
        self.startButton.clicked.connect(self.start_webcam)
        self.stopButton.clicked.connect(self.stop_webcam)
        self.detectButton.setCheckable(True)
        self.detectButton.toggled.connect(self.loadImage)
        self.kanji_Enabled=False
        self.mySlider.valueChanged.connect(self.changedValue)
        self.modeSlider.valueChanged.connect(self.chnageMode)


        self.setWindowTitle('Kanji detection')
        self.setWindowIcon(QIcon(os.path.join(ROOT_DIR, 'GUI/KanjiLogo.jpg')))
        self.show()

        self.app.exec_()
        # sys.exit()



    def loadImage(self):
        self.detectButton.setText('Load Image')
        fname = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\', "Image file(*.jpg *.png *.jpeg)")
        imagePath = fname[0]

        self.image = cv2.imread(imagePath)
        self.processed_image = self.process_frame(np.copy(self.image), self.roi_size, self.mode)
        cv2.rectangle(self.processed_image, (0, 0), (self.roi_size, self.roi_size), (0, 0, 255))

        self.resize(self.processed_image.shape[1], self.processed_image.shape[0])
        self.adjustSize()
        self.displayImage()


    def changedValue(self):
        self.roi_size = self.mySlider.value()
        self.label.setText(str(self.roi_size))
        self.processed_image = self.process_frame(np.copy(self.image), self.roi_size, self.mode)
        cv2.rectangle(self.processed_image, (0, 0), (self.roi_size, self.roi_size), (0, 0, 255))
        self.displayImage()

    def chnageMode(self):
        self.mode = self.modeSlider.value()
        self.label_2.setText(mode_dict[int(self.mode)])
        self.processed_image = self.process_frame(np.copy(self.image), self.roi_size, self.mode)
        cv2.rectangle(self.processed_image, (0, 0), (self.roi_size, self.roi_size), (0, 0, 255))
        self.displayImage()

    def start_webcam(self):
        try:

            self.capture = cv2.VideoCapture(0)  # 0 =default #1,2,3 =Extra Webcam
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

            ret, self.image = self.capture.read()
            if self.image is None:
                raise Exception

            self.cam_active = True

            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)
        except Exception:
            print("ERROR: no camera found.")  # TODO

    def update_frame(self):
        ret, self.image = self.capture.read()
        # self.image = cv2.flip(self.image, 1)

        self.processed_image = self.process_frame(np.copy(self.image), self.roi_size, self.mode)
        cv2.rectangle(self.processed_image, (0, 0), (self.roi_size, self.roi_size), (0, 0, 255))

        self.displayImage()

    def stop_webcam(self):
        if self.timer.isActive():
            self.timer.stop()

    def resize_window(self, new_img_size):
        old_w = self.imgLabel.width()
        old_h = self.imgLabel.height()
        new_h = (new_img_size[0] * old_w) // new_img_size[1]
        self.imgLabel.setFixedSize(old_w, new_h)
        height_offset = new_h - old_h
        width_offset = 0  # new_img_size[1] - old_w
        self.setFixedSize(
            self.width() + width_offset,
            self.height() + height_offset
        )
        move_ui_object(self.label, height_offset, 0)
        move_ui_object(self.label_2, height_offset, width_offset)
        move_ui_object(self.mySlider, height_offset, 0)
        move_ui_object(self.modeSlider, height_offset, width_offset)

    def displayImage(self):

        img = self.processed_image

        qformat = QImage.Format_Indexed8

        image_size = img.shape[:2]
        self.resize_window(image_size)

        if len(img.shape) == 3:  # rows[0],cols[1],channels[2]
            if (img.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        # BGR > RGB
        img = img.rgbSwapped()
        self.imgLabel.setPixmap(QPixmap.fromImage(img))
        self.imgLabel.setScaledContents(True)



if __name__ == '__main__':
    window = detect_kanji(lambda x: x)
