import sys

import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from PyQt5.uic import loadUi


class detect_kanji(QDialog):
    def __init__(self):
        super(detect_kanji, self).__init__()
        loadUi('detectkanji.ui', self)

        self.image = None
        self.startButton.clicked.connect(self.start_webcam)
        self.stopButton.clicked.connect(self.stop_webcam)
        self.detectButton.setCheckable(True)
        self.detectButton.toggled.connect(self.loadImage)
        self.kanji_Enabled=False
        self.mySlider.valueChanged.connect(self.changedValue)



    def loadImage(self):
        self.detectButton.setText('Load Image')
        fname = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\', "Image file(*.jpg *.png *.jpeg)")
        imagePath = fname[0]
        pixmap = QPixmap(imagePath)
        self.imgLabel.setPixmap(QPixmap(pixmap))
        self.resize(pixmap.width(),pixmap.height())
        self.adjustSize()
        self.show()


    def changedValue(self):
        size = self.mySlider.value()
        self.label.setText(str(size))


    def start_webcam(self):
        self.capture = cv2.VideoCapture(0)  # 0 =default #1,2,3 =Extra Webcam
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def update_frame(self):
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)

        if (self.kanji_Enabled):
            detected_image = self.detect_kanji(self.image)
            self.displayImage(detected_image, 1)
        else:
            self.displayImage(self.image, 1)

    def detect_kanji(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kanjies = self.kanjiCascade.detectMultiScale(gray, 1.2, 5, minSize=(90, 90))

        for (x, y, w, h) in kanjies:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        return img

    def stop_webcam(self):
        self.timer.stop()

    def displayImage(self, img, window=1):
        qformat = QImage.Format_Indexed8

        if len(img.shape) == 3:  # rows[0],cols[1],channels[2]
            if (img.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        # BGR > RGB
        img = img.rgbSwapped()
        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(img))
            self.imgLabel.setScaledContents(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = detect_kanji()
    window.setWindowTitle('Kanji detection')
    window.setWindowIcon(QIcon('KanjiLogo.jpg'))
    window.show()
    sys.exit(app.exec_())
