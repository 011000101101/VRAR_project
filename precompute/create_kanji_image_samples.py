# fonts from https://www.freejapanesefont.com/

import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2
import pickle
import os
from utils.params import *

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
    with open(os.path.join(ROOT_DIR, "resources/fonts/kanji_image_faulty_renderings.pkl"), 'rb') as f:
        faulty_imgs = pickle.load(f)

    # initialize dictionary to hold image samples indexed by kanji
    image_samples = dict()

    # load all available fonts
    fonts = []
    # use 3 different font sizes
    for i in range(0, 6, 2):

        print("creating samples for font size {}pt".format(SAMPLE_IMAGE_SIZE-i))

        # use each available font
        for font_path in FONT_PATHS:
            # complete font path
            font_path_absolute = os.path.join(ROOT_DIR, "resources/fonts/{}".format(font_path[1]))
            # load font
            font = ImageFont.truetype(font_path_absolute, SAMPLE_IMAGE_SIZE-i)
            fonts.append((font_path[0], font))

        kanji_count = 0

        # render each kanji...
        for kanji in kanji_list:

            if kanji_count%100 == 0:
                print("\tprocessed {} kanji.".format(kanji_count))
            kanji_count += 1

            image_samples_tmp = []

            # ...with each font
            for font in fonts:

                img = create_sample(kanji, font, i)

                # filter out not or faultily rendered kanji-font-combinations
                if (img == empty_img).all() or len([1 for faulty_img in faulty_imgs if (img == faulty_img).all()]) > 0:  # (img == faulty_img).all():
                    # print("character not supported: {}, font {}".format(kanji, font[0]))
                    pass
                else:
                    # convert image to greyscale to save space (semantically it is already in greyscale anyways, so just
                    # use different encodiing)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # TODO inspect samples
                    # cv2.imshow("asdf", img);cv2.waitKey();cv2.destroyAllWindows()

                    # accumulate rendered images
                    image_samples_tmp.append((font[0], img))

            # insert all images for a certain kanji into the dict
            try:
                image_samples[kanji]
                image_samples[kanji] += image_samples_tmp
            except KeyError:
                image_samples[kanji] = image_samples_tmp

    print("created {} clean samples ({} fonts, 3 sizes).".format(len(image_samples), len(FONT_PATHS)))

    # save on disk
    with open(os.path.join(ROOT_DIR, "bin_blobs/kanji_image_samples.pkl"), 'wb') as f:
        pickle.dump(image_samples, f, pickle.HIGHEST_PROTOCOL)


def create_sample(kanji: str, font, offset: int):
    # create empty white image
    img_tmp = empty_img.copy()
    # convert to correct format
    img_pil = Image.fromarray(img_tmp)
    # render single kanji by putting text onto the image
    draw = ImageDraw.Draw(img_pil)
    draw.text((offset/2, offset/2), kanji, font=font[1], fill=(b, g, r, a))
    # convert image back to numpy array
    img = np.array(img_pil)
    return img


def create_faulty_image_samples():
    faulty_samples = []

    font = ImageFont.truetype(os.path.join(ROOT_DIR, "resources/fonts/NikkyouSans/NikkyouSans-B6aV.ttf"), SAMPLE_IMAGE_SIZE)
    img = create_sample("阿", ("faulty", font), 0)
    faulty_samples.append(img)

    font = ImageFont.truetype(os.path.join(ROOT_DIR, "resources/fonts/NikkyouSans/NikkyouSans-B6aV.ttf"), SAMPLE_IMAGE_SIZE - 2)
    img = create_sample("阿", ("faulty", font), 2)
    faulty_samples.append(img)

    font = ImageFont.truetype(os.path.join(ROOT_DIR, "resources/fonts/NikkyouSans/NikkyouSans-B6aV.ttf"), SAMPLE_IMAGE_SIZE - 4)
    img = create_sample("阿", ("faulty", font), 4)
    faulty_samples.append(img)

    font = ImageFont.truetype(os.path.join(ROOT_DIR, "resources/fonts/MODI_komorebi-gothic_2018_0501/komorebi-gothic.ttf"), SAMPLE_IMAGE_SIZE)
    img = create_sample("黃", ("faulty", font), 0)
    faulty_samples.append(img)

    font = ImageFont.truetype(os.path.join(ROOT_DIR, "resources/fonts/MODI_komorebi-gothic_2018_0501/komorebi-gothic.ttf"), SAMPLE_IMAGE_SIZE - 2)
    img = create_sample("黃", ("faulty", font), 2)
    faulty_samples.append(img)

    font = ImageFont.truetype(os.path.join(ROOT_DIR, "resources/fonts/MODI_komorebi-gothic_2018_0501/komorebi-gothic.ttf"), SAMPLE_IMAGE_SIZE - 4)
    img = create_sample("黃", ("faulty", font), 4)
    faulty_samples.append(img)

    with open(os.path.join(ROOT_DIR, "resources/fonts/kanji_image_faulty_renderings.pkl"), 'wb') as f:
        pickle.dump(faulty_samples, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    create_faulty_image_samples()
    create_samples()
