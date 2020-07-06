import xml.etree.ElementTree as ET
import pickle


def hiraganify_single_on_yomi(on_yomi: str):

    utf8_bytes = list(on_yomi.encode("utf-8"))

    kun_yomi = ""

    for char in on_yomi:

        if char == 'ー':
            # new_char = kun_yomi[len(kun_yomi)-1]
            kun_yomi += 'あ'  # TODO dirty workaround for only occurence of 'ー' (ダース)
            continue

        # convert utf-9 string of single char to byte array holding its utf-8 codepoint
        char_bytes = list(char.encode("utf-8"))

        # skip non-katakana chars
        if char_bytes[0] != 227:
            continue

        ## assertion holds on full dataset
        # assert (
        #     (char_bytes[1] == 130 and 161 <= char_bytes[2] <= 191)
        #     or
        #     (char_bytes[1] == 131 and 128 <= char_bytes[2] <= 182)
        # ), "{} is not a katakana char: {}".format(char, char_bytes)  # 82a1 <= ... <= 83B6

        # change katakana char to equivalent hiragana char according to utf-8 codepoint table
        if char_bytes[1] == 130:  # 82
            char_bytes[1] = 129
            char_bytes[2] -= 32  # bf - 9f, distance of "ta"
        elif char_bytes[1] == 131:
            if char_bytes[2] < 160:  # a0
                char_bytes[1] = 129
                char_bytes[2] += 32  # 9f - bf, distance of "mi"
            else:
                char_bytes[1] = 130
                char_bytes[2] -= 32  # a0 - 80, distance of "mu"
        else:
            continue  # skip non-katakana chars

        # convert byte array holding utf-8 codepoint of single char back to utf-8 string
        new_char = bytes(char_bytes).decode("utf-8")

        # concatenate the characters
        kun_yomi += new_char

    return kun_yomi


def hiraganify_on_yomi(readings_on: list):
    return list(map(hiraganify_single_on_yomi, readings_on))


def isolate_actual_readings(readings_kun: list):
    return [extended_reading.split('.')[0] for extended_reading in readings_kun]


def cut_non_hiragana_chars(kun_yomi: str):

    utf8_bytes = list(kun_yomi.encode("utf-8"))

    hiragana_only = ""

    for char in kun_yomi:

        if char == 'ー':
            hiragana_only += 'ー'
            continue

        # convert utf-9 string of single char to byte array holding its utf-8 codepoint
        char_bytes = list(char.encode("utf-8"))

        # skip non-hiragana chars
        if char_bytes[0] != 227:
            continue

        # skip non-katakana chars
        if char_bytes[1] == 129:  # 82
            if not (129 <= char_bytes[2] <= 191):
                continue  # skip non-katakana chars
        elif char_bytes[1] == 130:
            if not (128 <= char_bytes[2] <= 150):
                continue  # skip non-katakana chars
        else:
            continue  # skip non-katakana chars

        # convert byte array holding utf-8 codepoint of single char back to utf-8 string
        new_char = bytes(char_bytes).decode("utf-8")

        # concatenate the characters
        hiragana_only += new_char

    return hiragana_only


def entry_list_to_map(entries_in: list):

    kanji_dict = {}

    for entry in entries_in:
        kanji = entry.find("literal").text
        readings_on = [reading.text for reading in entry.findall("reading_meaning/rmgroup/reading[@r_type='ja_on']")]
        readings_kun = [reading.text for reading in entry.findall("reading_meaning/rmgroup/reading[@r_type='ja_kun']")]

        readings_nanori = [reading.text for reading in entry.findall("reading_meaning/nanori")]

        readings = hiraganify_on_yomi(readings_on) + list(
            map(cut_non_hiragana_chars, isolate_actual_readings(readings_kun))
        )

        readings_nanori = list(map(cut_non_hiragana_chars, readings_nanori))

        kanji_dict[kanji] = (readings, readings_nanori)

    return kanji_dict


def convert():
    tree = ET.parse('../resources/kanjidic2.xml')

    entries = tree.findall("character")

    kanji_dict_map = entry_list_to_map(entries)

    with open("../bin_blobs/kanjidic2_hashtable.pkl", 'wb') as f:
        pickle.dump(kanji_dict_map, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    convert()
