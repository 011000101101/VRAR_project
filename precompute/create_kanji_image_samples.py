# fonts from https://www.freejapanesefont.com/

import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2
import pickle
import os
from utils.params import *

# specify names and paths of available fonts
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
    ("xano_mincho", "Xano-Mincho/XANO-mincho-U32.ttf")
]

# specify colour for font rendering
b, g, r, a = 0, 0, 0, 0  # totally black

# specify empty white image, which is computed once and then copied multiple times for performance reasons
empty_img = np.ones((SAMPLE_IMAGE_SIZE,SAMPLE_IMAGE_SIZE,3),np.uint8)
empty_img *= 255  # pure white background


def create_samples():
    """
    create image samples for each kanji: first load list of kanji from disk, then render each kanji using different
    fonts, finally save image samples to disk
    :return:
    """

    # load a list of all kanji to be rendered from the disk
    with open(os.path.join(ROOT_DIR, "bin_blobs/kanji_list.pkl"), 'rb') as f:
        kanji_list = pickle.load(f)

    # load an image of a faulty rendering to filter out unsupported kanji-font-combinations from disk
    # kanji unsupported by a certain font usually don't render at all, leaving an empty image, but sometimes they render
    # as a specific rectangle, identical to the image loaded here
    with open(os.path.join(ROOT_DIR, "resources/fonts/kanji_image_faulty_rendering.pkl"), 'rb') as f:
        faulty_img = pickle.load(f)

    # initialize dictionary to hold image samples indexed by kanji
    image_samples = dict()

    # load all available fonts
    fonts = []
    for font_path in font_paths:
        # complete font path
        font_path_absolute = os.path.join(ROOT_DIR, "resources/fonts/{}".format(font_path[1]))
        # load font
        font = ImageFont.truetype(font_path_absolute, SAMPLE_IMAGE_SIZE)
        fonts.append((font_path[0], font))

    # render each kanji...
    for kanji in kanji_list:

        image_samples_tmp = []

        # ...with each font
        for font in fonts:

            img = create_sample(kanji, font)

            # filter out not or faultily rendered kanji-font-combinations
            if (img == empty_img).all() or (img == faulty_img).all():
                # print("character not supported: {}, font {}".format(kanji, font[0]))
                pass
            else:
                # convert image to greyscale to save space (semantically it is already in greyscale anyways, so just
                # use different encodiing)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # cv2.imshow("asdf", img);cv2.waitKey();cv2.destroyAllWindows()

                # accumulate rendered images
                image_samples_tmp.append((font[0], img))

        # insert all images for a certain kanji into the dict
        image_samples[kanji] = image_samples_tmp

    # save on disk
    with open(os.path.join(ROOT_DIR, "bin_blobs/kanji_image_samples.pkl"), 'wb') as f:
        pickle.dump(image_samples, f, pickle.HIGHEST_PROTOCOL)


def create_sample(kanji: str, font):
    # create empty white image
    img_tmp = empty_img.copy()
    # convert to correct format
    img_pil = Image.fromarray(img_tmp)
    # render single kanji by putting text onto the image
    draw = ImageDraw.Draw(img_pil)
    draw.text((0, 0), kanji, font=font[1], fill=(b, g, r, a))
    # convert image back to numpy array
    img = np.array(img_pil)
    return img


def create_faulty_image_sample():
    font = ImageFont.truetype(os.path.join(ROOT_DIR, "resources/fonts/NikkyouSans/NikkyouSans-B6aV.ttf"), SAMPLE_IMAGE_SIZE)
    img = create_sample("阿", ("faulty", font))
    with open(os.path.join(ROOT_DIR, "resources/fonts/kanji_image_faulty_rendering.pkl"), 'wb') as f:
        pickle.dump(img, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    create_faulty_image_sample()
    create_samples()
