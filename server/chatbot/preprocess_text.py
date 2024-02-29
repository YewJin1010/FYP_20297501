from spellchecker import SpellChecker 

def autocorrect_text(text):
    spell = SpellChecker()
    misspelled = text.split()
    corrected_text = []
    for word in misspelled:
        corrected_text.append(spell.correction(word))
    print("corrected_text:", corrected_text)

    corrected_text = [word for word in corrected_text if word is not None]
    print("corrected_text:", corrected_text)
    if (len(corrected_text) == 0):
        return None
    else: 
        return " ".join(corrected_text)


