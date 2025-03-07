def get_unique_chars():
    # Czech uses Latin characters with diacritics (Latin Extended-A block)
    czech_chars = [chr(i) for i in range(0x0100, 0x0180) if chr(i).isprintable()]  # Czech (Latin Extended-A)

    # Greek alphabet (Greek and Coptic Unicode block)
    greek_chars = [chr(i) for i in range(0x0370, 0x0400) if chr(i).isprintable()]  # Greek

    # Hebrew alphabet (Hebrew Unicode block)
    hebrew_chars = [chr(i) for i in range(0x0590, 0x0600) if chr(i).isprintable()]  # Hebrew

    # Russian alphabet (Cyrillic Unicode block)
    russian_chars = [chr(i) for i in range(0x0400, 0x0500) if chr(i).isprintable()]  # Russian (Cyrillic)

    # Arabic alphabet (Arabic Unicode block)
    arabic_chars = [chr(i) for i in range(0x0600, 0x0700) if chr(i).isprintable()]  # Arabic

    # Korean characters (Hangul Syllables Unicode block)
    korean_chars = [chr(i) for i in range(0xAC00, 0xD7A4) if chr(i).isprintable()]  # Korean (Hangul syllables)

    # Macedonian alphabet (using the Cyrillic Supplement block for uniqueness)
    macedonian_chars = [chr(i) for i in range(0x0500, 0x0530) if chr(i).isprintable()]  # Macedonian (Cyrillic Supplement)

    # Thai characters (Thai Unicode block)
    thai_chars = [chr(i) for i in range(0x0E00, 0x0E80) if chr(i).isprintable()]  # Thai

    # Hindi characters (Devanagari Unicode block)
    hindi_chars = [chr(i) for i in range(0x0900, 0x0980) if chr(i).isprintable()]  # Hindi (Devanagari)

    # Bengali characters (Bengali Unicode block)
    bengali_chars = [chr(i) for i in range(0x0980, 0x0A00) if chr(i).isprintable()]  # Bengali

    """
    print("Czech:", "".join(czech_chars))
    print("Greek:", "".join(greek_chars))
    print("Hebrew:", "".join(hebrew_chars))
    print("Russian:", "".join(russian_chars))
    print("Arabic:", "".join(arabic_chars))
    print("Korean:", "".join(korean_chars[:50]))   # output only first 50 for brevity
    print("Macedonian:", "".join(macedonian_chars))
    print("Thai:", "".join(thai_chars))
    print("Hindi:", "".join(hindi_chars))
    print("Bengali:", "".join(bengali_chars))
    """

    latin_chars = [chr(i) for i in range(0x0020, 0x007B) if chr(i).isprintable()]  # Basic Latin (A-Z, a-z)
    chinese_chars = [chr(i) for i in range(0x4E00, 0x9FFF) if chr(i).isprintable()]  # Common Chinese characters

    # French uses Latin characters with accents
    french_chars = [chr(i) for i in range(0x00C0, 0x0100) if chr(i).isprintable()]  # À-ÿ (includes é, è, ç, etc.)

    # Japanese includes Hiragana, Katakana
    hiragana_chars = [chr(i) for i in range(0x3041, 0x30A0) if chr(i).isprintable()]  # Hiragana
    katakana_chars = [chr(i) for i in range(0x30A1, 0x3100) if chr(i).isprintable()]  # Katakana



    # Combine all characters and verify uniqueness
    unique_chars = set("".join(
        czech_chars + greek_chars + hebrew_chars + russian_chars + 
        arabic_chars + korean_chars + macedonian_chars + thai_chars + 
        hindi_chars + bengali_chars + latin_chars + chinese_chars + 
        french_chars + hiragana_chars + katakana_chars
    ))
    """
    print(len(unique_chars))
    print(len(unique_chars) == len(czech_chars) + len(greek_chars) + len(hebrew_chars) + 
        len(russian_chars) + len(arabic_chars) + len(korean_chars) + len(macedonian_chars) + 
        len(thai_chars) + len(hindi_chars) + len(bengali_chars) + len(latin_chars) + 
        len(chinese_chars) + len(french_chars) + len(hiragana_chars) + len(katakana_chars))
    """

    # manually extra
    unique_chars.add("¦")
    unique_chars.add('\u3000')
    unique_chars.add('、')
    unique_chars.add('。')
    unique_chars.add('\uff1f')
    unique_chars.add('\uff01')
    unique_chars.add('「')
    unique_chars.add('」')
    unique_chars.add('『')
    unique_chars.add('』')
    unique_chars.add('《')
    unique_chars.add('》')
    unique_chars.add('\uff0d')
    unique_chars.add('\uff10')
    unique_chars.add('々')

    # Define large Unicode character mapping
    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    num_classes = len(unique_chars)
    # print(num_classes)
    return idx_to_char, num_classes