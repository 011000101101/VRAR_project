import xml.etree.ElementTree as ET
import pickle


def convert():
    """
    loads the kanjidic2 file from disk, converts it to a list of the commonly used kanji (groups G1 - G10), and saves
    it as a binary blob
    :return:
    """

    # load kanjidic2 file as an element tree
    tree = ET.parse('../resources/kanjidic2.xml')

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

    # save on disk
    with open("../bin_blobs/kanji_list.pkl", 'wb') as f:
        pickle.dump(kanji_list, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    convert()
