import precompute.convert_JMdict_to_hashtable as convert_JMdict_to_hashtable
import precompute.convert_kanjidic2_to_hashtable as convert_kanjidic2_to_hashtable
import precompute.convert_kanjidic2_to_list as convert_kanjidic2_to_list
import precompute.create_kanji_image_samples as create_kanji_image_samples

if __name__ == "__main__":
    convert_JMdict_to_hashtable.convert()
    convert_kanjidic2_to_hashtable.convert()
    convert_kanjidic2_to_list.convert()
    create_kanji_image_samples.create_samples()
