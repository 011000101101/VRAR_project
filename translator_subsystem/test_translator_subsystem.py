import numpy as np

import translator_subsystem.lut_translator as translator_subsystem


def translate_and_mask_full_sequence_for_testing(sequence: str, n_leading_kanji: int):
    partial_translations = translator_subsystem.translate_and_mask_sequence(sequence, n_leading_kanji)
    translation = "".join([partial_translation[2] for partial_translation in partial_translations])
    used_chars = np.sum([[partial_translation[2] for partial_translation in partial_translations]])
    translation += sequence[used_chars:]
    return translation


def run_single_test(test_input: tuple, expected_outputs: list):
    translation = translate_and_mask_full_sequence_for_testing(*test_input)
    assert translation in expected_outputs, \
        "Test failed: trying to translate '{}': expected one of '{}' but got '{}'".format(
            test_input[0], expected_outputs, translation
        )


def run_batch_test(test_data_set: list, test_masking_set: set):
    translator_subsystem.overwrite_masked_kanji_set(test_masking_set)

    for test_data_instance in test_data_set:
        run_single_test(*test_data_instance)

    translator_subsystem.reset_masked_kanji_set()


def test_manga_pane_no_masking():
    test_data_set = [
        (("七英雄の", 3), ["ななえいゆうの", "しちえいゆうの"]),  # both valid readings
        (("一人ぐれん", 2), ["ひとりぐれん"]),
        (("侯の", 1), ["こうの"]),
        (("使者である", 2), ["ししゃである"]),
        (("貴方達が", 3), ["あなたたちが"]),
        (("反逆者どもの", 3), ["はんぎゃくしゃどもの"]),
        (("暴虐に", 2), ["ぼうぎゃくに"]),
        (("苦しむ", 1), ["くるしむ"]),
        (("我ら", 1), ["われら"]),
        (("辺境の", 2), ["へんきょうの"]),
        (("帝臣から", 2), ["ていしんから", "みかどおみから"])  # using jisho.org as baseline, 帝臣 does not have to be
        # translated to ていしん, but individual translation (帝 -> みかど, 臣 -> おみ) suffices
    ]
    run_batch_test(test_data_set, set())


def test_manga_pane_with_masking():
    test_masking_set = set()
    test_masking_set.add('七')
    test_masking_set.add('雄')
    test_masking_set.add('一')
    test_masking_set.add('者')
    test_masking_set.add('方')
    test_masking_set.add('苦')
    test_masking_set.add('辺')
    test_data_set = [
        (("七英雄の", 3), ["〇えい〇の"]),
        (("一人ぐれん", 2), ["〇りぐれん"]),
        (("侯の", 1), ["こうの"]),
        (("使者である", 2), ["し〇である"]),
        (("貴方達が", 3), ["あなたたちが"]),  # anata is not composed from the readings of its kanji,
        # "independent composite reading"
        (("反逆者どもの", 3), ["はんぎゃく〇どもの"]),
        (("暴虐に", 2), ["ぼうぎゃくに"]),
        (("苦しむ", 1), ["〇しむ"]),
        (("我ら", 1), ["われら"]),
        (("辺境の", 2), ["〇きょうの"]),
        (("帝臣から", 2), ["ていしんから", "みかどおみから"])  # using jisho.org as baseline, 帝臣 does not have to be
        # translated to ていしん, but individual translation (帝 -> みかど, 臣 -> おみ) suffices
    ]
    run_batch_test(test_data_set, test_masking_set)
