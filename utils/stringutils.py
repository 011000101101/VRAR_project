def is_kana(char: str):
    # convert utf-9 string of single char to byte array holding its utf-8 codepoint
    char_bytes = list(char.encode("utf-8"))

    # skip non-hiragana chars  e3 81 81 - e3 82 9f
    if char_bytes[0] == 227:

        if char_bytes[1] == 129:  # 81

            if 129 <= char_bytes[2] <= 191:  # 81 - bf

                return True  # is Hiragana

        elif char_bytes[1] == 130:

            if 128 <= char_bytes[2] <= 150:  # 80 - 96

                return True  # is Hiragana

            elif 153 <= char_bytes[2] <= 191:  # 99 - bf

                return True  # is Katakana

        elif char_bytes[1] == 131:

            if 128 <= char_bytes[2] <= 191:  # 80 - bf

                return True  # is Katakana

    return False  # not a Kana char
