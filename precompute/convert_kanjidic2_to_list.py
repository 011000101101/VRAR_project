import xml.etree.ElementTree as ET
import pickle
import os
from utils.params import *
import numpy as np


def convert():
    """
    loads the kanjidic2 file from disk, converts it to a list of the commonly used kanji (groups G1 - G10), and saves
    it as a binary blob
    :return:
    """

    # load kanjidic2 file as an element tree
    tree = ET.parse(os.path.join(ROOT_DIR, "resources/kanjidic2/kanjidic2.xml"))

    # find all kanji entries
    entries = tree.findall("character")

    # convert list of Elements to list of strings containing one kanji each
    kanji_list = [
        entry.find("literal").text
        for
        entry
        in
        entries
        if
        entry.find("misc/grade")is not None and int(entry.find("misc/grade").text) <= 10
    ]

    print(len(kanji_list))
    return kanji_list


def add_kana(kanji_list: list) -> list:

    char_bytes = np.empty(3, "uint8")

    char_bytes[0] = 227

    char_bytes[1] = 129  # 81

    for i in range(129, 192):  # 81 - bf

        char_bytes[2] = i

        kanji_list.append(bytes(char_bytes).decode("utf-8"))

    char_bytes[1] = 130

    for i in range(128, 149):  # 80 - 96

        char_bytes[2] = i

        kanji_list.append(bytes(char_bytes).decode("utf-8"))

    for i in range(161, 192):  # 99 - bf

        char_bytes[2] = i

        kanji_list.append(bytes(char_bytes).decode("utf-8"))

    char_bytes[1] = 131

    for i in range(128, 181):  # 80 - bf

        char_bytes[2] = i

        kanji_list.append(bytes(char_bytes).decode("utf-8"))

    return kanji_list


def save_kanji_list(kanji_list: list):

    # save on disk
    with open(os.path.join(ROOT_DIR, "bin_blobs/kanji_list.pkl"), 'wb') as f:
        pickle.dump(kanji_list, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    kanji_list_tmp = convert()
    kanji_list_tmp = add_kana(kanji_list_tmp)
    save_kanji_list(kanji_list_tmp)
