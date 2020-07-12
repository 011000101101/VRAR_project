import xml.etree.ElementTree as ET
import pickle
import os
from utils.params import *


def skip_inflection(keb: str, reb: str):
    return []


def inflect_ichidan_verb(keb: str, reb: str):
    new_endings = ["ま", "な", "た", "られ", "よう", "て"]
    inflected_group = []
    keb_tmp = keb[:-1]
    reb_tmp = reb[:-1]
    for new_ending in new_endings:
        inflected_group.append((keb_tmp + new_ending, reb_tmp + new_ending))
    return inflected_group


def inflect_kuru_special_verb(keb: str, reb: str):
    new_endings = ["きま", "こな", "きた", "こられ", "よう", "きて", "こい"]
    inflected_group = []
    keb_tmp = keb[:-1]
    reb_tmp = reb[:-2]
    for new_ending in new_endings:
        inflected_group.append((keb_tmp + new_ending[1:], reb_tmp + new_ending))
    return inflected_group


def inflect_godan_verb(keb: str, reb: str, ending: str):
    inflection_stems = {
        "ぶ": ["ば", "び", "べ", "ぼ", "んで"],
        "ぐ": ["が", "ぎ", "げ", "ご", "いで"],
        "く": ["か", "き", "け", "こ", "いて"],
        "む": ["ま", "み", "め", "も", "んで"],
        "ぬ": ["な", "に", "ね", "の", "んで"],
        "る": ["ら", "り", "れ", "ろ", "って"],
        "す": ["さ", "し", "せ", "そ", "して"],
        "つ": ["た", "ち", "て", "と", "って"],
        "う": ["あ", "い", "え", "お", "って"],
    }
    new_endings = [
        ["な"],  # new_endings_a: nai
        ["ま", "た", "な"],  # new_endings_i: masu, tai, nasai
        ["ま", "な", ""],  # new_endings_e: masu, nai, [imperative]
        ["う"],  # new_endings_o: u
        [""]  # te
    ]
    inflected_group = []
    for i in range(len(inflection_stems[ending])):
        keb_tmp = keb[:-1] + inflection_stems[ending][i]
        reb_tmp = reb[:-1] + inflection_stems[ending][i]
        for new_ending in new_endings[i]:
            inflected_group.append((keb_tmp + new_ending, reb_tmp + new_ending))
    return inflected_group


def inflect_godan_bu_verb(keb: str, reb: str):
    return inflect_godan_verb(keb, reb, "ぶ")


def inflect_godan_gu_verb(keb: str, reb: str):
    return inflect_godan_verb(keb, reb, "ぐ")


def inflect_godan_ku_verb(keb: str, reb: str):
    return inflect_godan_verb(keb, reb, "く")


def inflect_godan_mu_verb(keb: str, reb: str):
    return inflect_godan_verb(keb, reb, "む")


def inflect_godan_nu_verb(keb: str, reb: str):
    return inflect_godan_verb(keb, reb, "ぬ")


def inflect_godan_ru_verb(keb: str, reb: str):
    return inflect_godan_verb(keb, reb, "る")


def inflect_godan_su_verb(keb: str, reb: str):
    return inflect_godan_verb(keb, reb, "す")


def inflect_godan_tsu_verb(keb: str, reb: str):
    return inflect_godan_verb(keb, reb, "つ")


def inflect_godan_u_verb(keb: str, reb: str):
    return inflect_godan_verb(keb, reb, "う")


def produce_all_inflections(entry: ET.Element, keb: str, reb: str):
    switcher = {
        "Ichidan verb": inflect_ichidan_verb,
        "Ichidan verb - kureru special class": skip_inflection,
        "Nidan verb with 'u' ending (archaic)": skip_inflection,
        "Yodan verb with `hu/fu' ending (archaic)": skip_inflection,
        "Yodan verb with `ru' ending (archaic)": skip_inflection,
        "Godan verb - -aru special class": skip_inflection,
        "Godan verb with `bu' ending": inflect_godan_bu_verb,
        "Godan verb with `gu' ending": inflect_godan_gu_verb,
        "Godan verb with `ku' ending": inflect_godan_ku_verb,
        "Godan verb - Iku/Yuku special class": skip_inflection,
        "Godan verb with `mu' ending": inflect_godan_mu_verb,
        "Godan verb with `nu' ending": inflect_godan_nu_verb,
        "Godan verb with `ru' ending": inflect_godan_ru_verb,
        "Godan verb with `ru' ending (irregular verb)": skip_inflection,
        "Godan verb with `su' ending": inflect_godan_su_verb,
        "Godan verb with `tsu' ending": inflect_godan_tsu_verb,
        "Godan verb with `u' ending": inflect_godan_u_verb,
        "Godan verb with `u' ending (special class)": skip_inflection,
        "Godan verb - Uru old class verb (old form of Eru)": skip_inflection,
        "Ichidan verb - zuru verb (alternative form of -jiru verbs)": skip_inflection,
        "intransitive verb": skip_inflection,
        "Kuru verb - special class": inflect_kuru_special_verb,
        "irregular nu verb": skip_inflection,
        "irregular ru verb, plain form ends with -ri": skip_inflection,
        "noun or participle which takes the aux. verb suru": skip_inflection,
        "su verb - precursor to the modern suru": skip_inflection,
        "suru verb - special class": skip_inflection,
        "suru verb - included": skip_inflection
    }
    inflected_group = [(keb, reb)]
    all_pos = entry.findall("sense/pos")
    for pos in all_pos:
        if pos is not None:
            inflect_function = switcher.get(pos.text, skip_inflection)
            inflected_group += inflect_function(keb, reb)
    return inflected_group


def entry_list_to_map(entries_in: list):

    word_dict = {}

    # for entry in entries_in:
    #     for keb in entry.findall("k_ele/keb"):
    #         kanji_dict[keb.text] = []
    #         all_inflections[keb.text] = produce_all_inflections(entry, keb.text)
    #
    # for entry in entries_in:
    #     readings = [reb.text for reb in entry.findall("r_ele/reb")]
    #     for keb in entry.findall("k_ele/keb"):
    #         kanji_dict[keb.text] += readings

    for entry in entries_in:
        readings = entry.findall("r_ele")
        for keb in entry.findall("k_ele/keb"):

            # TODO trying to skip duplicates
            try:
                word_dict[keb.text]
                continue
            except KeyError:
                pass

            applicable_reading = ""
            for reading in readings:
                applicable_reading = reading.find("reb").text
                for re_restr in reading.findall("re_restr"):
                    applicable_reading = ""
                    if re_restr.text == keb.text:
                        applicable_reading = reading.find("reb").text
                        break
                if applicable_reading != "":
                    break

            all_inflections = produce_all_inflections(entry, keb.text, applicable_reading)
            for inflection in all_inflections:
                word_dict[inflection[0]] = inflection[1]

    return word_dict


def convert():
    tree = ET.parse(os.path.join(ROOT_DIR, "resources/JMdict/JMdict_e"))

    entries = tree.findall("entry/k_ele/...")

    word_dict_map = entry_list_to_map(entries)

    with open(os.path.join(ROOT_DIR, "bin_blobs/JMdict_e_hashtable.pkl"), 'wb') as f:
        pickle.dump(word_dict_map, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    convert()
