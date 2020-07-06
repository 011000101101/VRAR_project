# fonts from https://www.freejapanesefont.com/

import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2
import pickle

with open("../bin_blobs/kanji_list.pkl", 'rb') as f:
    kanji_list = pickle.load(f)

with open("../bin_blobs/faulty_kanji.pkl", 'rb') as f:
    faulty_img = pickle.load(f)

FONT_SIZE_PT = 48
font_paths = [
    ("komorebi_gothic", "MODI_komorebi-gothic_2018_0501/komorebi-gothic.ttf"),
    ("kosugi_maru", "Kosugi_Maru/KosugiMaru-Regular.ttf"),
    ("chi_jyun", "chi_jyun/é┐é═éÔÅâ.ttf"),
    ("togalite-light", "togalite/togalite-light.otf"),
    ("togalite-regular", "togalite/togalite-regular.otf"),
    ("togalite-medium", "togalite/togalite-medium.otf"),
    ("togalite-bold", "togalite/togalite-bold.otf"),
    ("togalite-heavy", "togalite/togalite-heavy.otf"),
    ("togalite-black", "togalite/togalite-black.otf"),
    ("chikara_dzuyoku", "chikara_dzuyoku/851CHIKARA-DZUYOKU_kanaA_004.ttf"),
    ("nikkyou_sans", "NikkyouSans/NikkyouSans-B6aV.ttf"),
    ("xano_mincho", "00/Xano-Mincho/XANO-mincho-U32.ttf")
]
b, g, r, a = 0, 0, 0, 0  # totally black
empty_img = np.ones((50,50,3),np.uint8)
empty_img *= 255  # pure white background


def create_samples():

    image_samples = dict()

    fonts = []
    for font_path in font_paths:
        font_path_absolute = "../resources/fonts/{}".format(font_path[1])
        font = ImageFont.truetype(font_path_absolute, FONT_SIZE_PT)
        fonts.append((font_path[0], font))

    for kanji in kanji_list:

        try:
            image_samples[kanji]
        except KeyError:
            image_samples_tmp = []

            for font in fonts:

                img_tmp = empty_img.copy()
                img_pil = Image.fromarray(img_tmp)
                draw = ImageDraw.Draw(img_pil)
                draw.text((0, 0), kanji, font=font[1], fill=(b, g, r, a))
                img = np.array(img_pil)

                if (img == empty_img).all() or (img == faulty_img).all():
                    # print("character not supported: {}, font {}".format(kanji, font[0]))
                    pass
                else:
                    # cv2.imshow("asdf", img);cv2.waitKey();cv2.destroyAllWindows()
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    image_samples_tmp.append((font[0], img))

            image_samples[kanji] = image_samples_tmp

    with open("../bin_blobs/kanji_image_samples.pkl", 'wb') as f:
        pickle.dump(image_samples, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    create_samples()
