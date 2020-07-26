import pickle
import utils.stringutils as stringutils
import os
from utils.params import *

with open(os.path.join(ROOT_DIR, "bin_blobs/JMdict_e_hashtable.pkl"), 'rb') as f:
    word_dict = pickle.load(f)

with open(os.path.join(ROOT_DIR, "bin_blobs/kanjidic2_hashtable.pkl"), 'rb') as f:
    kanji_dict = pickle.load(f)

with open(os.path.join(ROOT_DIR, "translator_subsystem/masked_kanji.pkl"), 'rb') as f:
    masked_kanji_set = pickle.load(f)


class NoValidCombinationOfReadingsFoundError(Exception):
    pass


def find_exact_match(keb: str):
    return word_dict[keb]


def translate_sequence_recursively(sequence: str, n_leading_kanji: int):
    # print("trying to translate {}".format(sequence))
    if sequence == "" or n_leading_kanji <= 0:
        return [(len(sequence), sequence, sequence)]
    current_len = len(sequence)
    while current_len > 0:
        try:
            reading = find_exact_match(sequence[:current_len])
            remaining_readings = translate_sequence_recursively(sequence[current_len:], n_leading_kanji - current_len)
            remaining_readings.append((current_len, sequence[:current_len], reading))
            return remaining_readings
        except KeyError:
            current_len -= 1
    raise KeyError


def translate_sequence(sequence: str, n_leading_kanji: int):
    current_len = len(sequence)
    while current_len > 0:
        try:
            translation = translate_sequence_recursively(sequence[:current_len], min(n_leading_kanji, current_len))
            translation.reverse()
            # translation = translation[:-1]
            return current_len, translation
        except KeyError:
            current_len -= 1
    return 0, []


def requires_masking(kanji: str):
    return kanji in masked_kanji_set


def find_masking_positions(keb: str):
    masking_positions = []
    for i_tmp in range(len(keb)):
        if requires_masking(keb[i_tmp]):
            masking_positions.append(i_tmp)
    return masking_positions


def segment_reading_recursively(reading: str, keb: str, include_name_readings: bool = False):

    # end of recursion reached, create list and pass upward if reading characters were used up,
    # raise exception otherwise
    if keb == "":
        if reading == "":
            return []
        else:
            raise NoValidCombinationOfReadingsFoundError

    # process the last kanji in keb
    kanji = keb[len(keb)-1]
    # get all readings
    readings, readings_nanori = kanji_dict[kanji]
    # if name readings are to be included, add them to the rear of the list of normal readings
    if include_name_readings:
        readings += readings_nanori
    # find a matching reading that allows for the rest of the word to still be segmented correctly
    for partial_reading in readings:
        # if this reading of the kanji fits the end of the current portion of the word, try it
        if reading.endswith(partial_reading):
            try:
                # try to segment the remainder of the word
                segmented_reading = segment_reading_recursively(
                    reading[:-len(partial_reading)], keb[:-1], include_name_readings=include_name_readings
                )
                # if successful, append this partial_reading and return
                segmented_reading.append(partial_reading)
                return segmented_reading
            # if the rest of the reading was not correctly segmentable, pass this partial_reading and try the next one
            except NoValidCombinationOfReadingsFoundError:
                continue

    # none of the readings for this kanji fitted or allowed for the remainder of the word to be segmented,
    # pass exception upward
    raise NoValidCombinationOfReadingsFoundError


def segment_reading(reading: str, keb: str, n_leading_kanji: int, include_name_readings: bool = False):
    if n_leading_kanji < len(keb):
        number_of_trailing_hiragana = len(keb) - n_leading_kanji  # >= 1
        segmented_reading = segment_reading_recursively(
            reading[:-number_of_trailing_hiragana], keb[:-number_of_trailing_hiragana],
            include_name_readings=include_name_readings
        )
        # if successful, append this partial_reading and return
        segmented_reading.append(reading[-number_of_trailing_hiragana:])
        return segmented_reading
    else:
        return segment_reading_recursively(reading, keb, include_name_readings=include_name_readings)


def mask_word(reading: str, keb: str, n_leading_kanji: int):
    masking_positions = find_masking_positions(keb)
    if not masking_positions:  # eq. masking_positions == []
        return reading
    try:
        segmented_reading = segment_reading(reading, keb, n_leading_kanji)
    except NoValidCombinationOfReadingsFoundError:
        segmented_reading = segment_reading(reading, keb, n_leading_kanji, include_name_readings=True)

    # mask reading of hidden kanji
    for position in masking_positions:
        # "maru" for censored character. note that one '〇' can cover several kana, but always exactly one kanji
        segmented_reading[position] = "〇"

    # join modified segmented reading into single string
    masked_reading = ""
    for partial_reading in segmented_reading:
        masked_reading += partial_reading

    return masked_reading


def translate_and_mask_sequence(sequence: str, n_leading_kanji: int):
    used_chars, translated_sequence = translate_sequence(sequence=sequence, n_leading_kanji=n_leading_kanji)
    masked_sequence = []
    n_leading_lanji_tmp = n_leading_kanji
    for word in translated_sequence:
        n_leading_kanji_local = min(len(word[1]), n_leading_lanji_tmp)
        try:
            masked_reading = mask_word(reading=word[2], keb=word[1], n_leading_kanji=n_leading_kanji_local)
        except NoValidCombinationOfReadingsFoundError:  # if no valid combination of basic readings found...
            if all(requires_masking(kanji) for kanji in word[1][:n_leading_kanji_local]):
                masked_reading = "".join(["〇" for _ in range(n_leading_kanji_local)])
            else:
                masked_reading = word[2]
        # reduce leading kanji count so it representes the number of leading kanji in the remaining sequence
        n_leading_lanji_tmp -= min(len(word[1]), n_leading_lanji_tmp)
        masked_sequence.append((word[0], word[1], masked_reading))
    return masked_sequence  # used_chars, masked_sequence


def translate_and_mask_line(line: str):
    """
    translate and mask a whole line (several concatenated sequences)
    :param line: the line to translate
    :return: the list of translated and masked sequences, each with their length (char count) and leading kanji count
    """
    current_sequence_start = 0
    last_char_was_kana = False
    translated_and_masked_sequences = []
    number_of_kanji = 0
    # process each char
    for i in range(len(line)):
        # if current char is kana, mark it as such and continue
        if stringutils.is_kana(line[i]):
            last_char_was_kana = True
        # if it is kanji...
        else:
            # ... and the last char was kana, this is the beginning of a new sequence, and the old/finished sequence
            # can be processed and saved
            if last_char_was_kana:
                # translate this sequence and save it
                translated_and_masked_sequences.append(
                    (
                        translate_and_mask_sequence(line[current_sequence_start:i], number_of_kanji),
                        i - current_sequence_start,
                        number_of_kanji
                    )
                )
                # reset kanji counter, set start index of next sequence
                current_sequence_start = i
                number_of_kanji = 1
            # ... otherwise, just count the kanji
            else:
                number_of_kanji += 1
            # and mark this char as kanji for the next iteration
            last_char_was_kana = False
    # translate the last sequence and save it
    translated_and_masked_sequences.append(
        (
            translate_and_mask_sequence(line[current_sequence_start:], number_of_kanji),
            len(line) - current_sequence_start,
            number_of_kanji
        )
    )
    return translated_and_masked_sequences



def overwrite_masked_kanji_set(new_set: set):
    global masked_kanji_set
    masked_kanji_set = new_set


def reset_masked_kanji_set():
    global masked_kanji_set
    with open(os.path.join(ROOT_DIR, "translator_subsystem/masked_kanji.pkl"), 'rb') as f:
        masked_kanji_set = pickle.load(f)


if __name__ == "__main__":

    print(translate_and_mask_sequence("日本語", 3))
