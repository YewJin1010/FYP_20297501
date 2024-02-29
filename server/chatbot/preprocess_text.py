from nltk.tokenize import word_tokenize
from nltk.corpus import words
from spellchecker import SpellChecker 

def autocorrect_text(text):
    spell = SpellChecker()
    misspelled = text.split()
    corrected_text = []
    for word in misspelled:
        corrected_text.append(spell.correction(word))
    print("corrected_text:", corrected_text)
    if (corrected_text == [None]): 
        return None
    else: 
        return " ".join(corrected_text)

def remove_gibberish(text):
    is_words = []
    # Tokenize the text
    tokens = word_tokenize(text)
    print(tokens)
 
    # Check for repeat characters
    for token in tokens:
        if token in words.words():
            is_words.append(token)
    
    print(is_words)
    return " ".join(is_words)


def preprocess_text(text):
    text = autocorrect_text(text)
    if text is None:
        return None
    else: 
        text = remove_gibberish(text)
        return text
