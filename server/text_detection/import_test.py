import nltk
import os

def nltk_stopwords_location():
    # Get the NLTK data directory
    nltk_data_dir = nltk.data.find('.')

    # Check if the stopwords file exists
    stopwords_file = os.path.join(nltk_data_dir, 'corpora/stopwords/english')
    if os.path.exists(stopwords_file):
        return stopwords_file
    else:
        return "NLTK stopwords file not found."

# Check the location of NLTK stopwords
print(nltk_stopwords_location())
