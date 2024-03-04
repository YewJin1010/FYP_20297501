import pandas as pd
import numpy as np
import re
import ast
import string
from functools import reduce
from operator import add
import spacy
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA, NMF
import time
import networkx as nx
import scipy
import os
from itertools import repeat

# spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

def index_categories(data, column):
    return [index for i, index in zip(data[column], data.index) if all(j.isdigit() or j in string.punctuation for j in i)]

def drop_rows_by_indices(data, indices):
    return data.drop(index=set(reduce(add, indices))).reset_index(drop=True)

def process_recipes(recipes):
    # Drop recipe instructions with less than 20 characters
    empty_instr_ind = [index for i, index in zip(recipes['directions'], recipes.index) if len(i) < 20]
    recipes = recipes.drop(index=empty_instr_ind).reset_index(drop=True)

    # Convert string representations of lists to actual lists for all recipes
    recipes['ingredients'] = recipes['ingredients'].apply(ast.literal_eval)

    # Extracting ingredients from their lists and formatting as single strings
    recipes['ingredient_text'] = ['; '.join(ingredients) for ingredients in recipes['ingredients']]

    # Counting the number of ingredients used in each recipe
    recipes['ingredient_count'] = [len(ingredients) for ingredients in recipes['ingredients']]

    # Convert string representations of lists to actual lists for all recipes
    recipes['directions'] = recipes['directions'].apply(ast.literal_eval)

    # Counting the number of directions in each recipe
    recipes['directions_count'] = [len(instruction) for instruction in recipes['directions']]

    # Formatting directions as single strings
    recipes['directions'] = [' '.join(instruction[1:]).strip('[]') if index > 0 else ' '.join(instruction).strip('[]') for index, instruction in enumerate(recipes['directions'])]

    return recipes

def clean_text(documents):
    cleaned_text = []
    for doc in documents:
        doc = doc.translate(str.maketrans('', '', string.punctuation)) # Remove Punctuation
        doc = re.sub(r'\d+', '', doc) # Remove Digits
        doc = doc.replace('\n',' ') # Remove New Lines
        doc = doc.strip() # Remove Leading White Space
        doc = re.sub(' +', ' ', doc) # Remove multiple white spaces
        cleaned_text.append(doc)
    return cleaned_text

# Tokenizing Function that lemmatizes words and removes Stop Words
def text_tokenizer(documents):
    tokenized_documents = [
        ' '.join([token.lemma_ for token in nlp(doc) if not token.is_stop])
        for doc in documents
    ]
    return tokenized_documents

# Tokenizing Using Spacy and run in parallel
def text_tokenizer_mp(doc):
    return ' '.join([token.lemma_ for token in nlp(doc) if not token.is_stop])

# Function to generate Word Embeddings
def generate_word_embeddings(all_text):
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 1))
    text_tfidf = vectorizer.fit_transform(all_text)
    tfidf_words = vectorizer.get_feature_names_out()
    return text_tfidf, tfidf_words

def docs_by_tops(top_mat, topic_range = (0,0), doc_range = (0,2)):
    for i in range(topic_range[0], topic_range[1]):
        topic_scores = pd.Series(top_mat[:,i])
        doc_index = topic_scores.sort_values(ascending = False)[doc_range[0]:doc_range[1]].index
        for j, index in enumerate(doc_index, doc_range[0]):
            print('Topic #{}'.format(i),
                  '\nDocument #{}'.format(j),
                  '\nTopic Score: {}\n\n'.format(topic_scores[index]),
                  text_series[index], '\n\n')

# Function for best topic words using cosine similarity
def words_by_tops(tfidf_mat, top_mat, topic_range=(0,0), n_words=10):
    topic_word_scores = tfidf_mat.T * top_mat
    for i in range(topic_range[0],topic_range[1]):
        word_scores = pd.Series(topic_word_scores[:,i])
        word_index = word_scores.sort_values(ascending = False)[:n_words].index
        print('\nTopic #{}'.format(i))
        for index in word_index:
            print(word_series[index],'\t\t', word_scores[index])
                  
def topic_docs_4kwsummary(topic_document_scores, root_text_data):
    '''Gathers and formats the top recipes in each topic'''
    text_index = pd.Series(topic_document_scores).sort_values(ascending = False)[:N_top_docs].index
    text_4kwsummary = pd.Series(root_text_data)[text_index]
    return text_4kwsummary

def generate_filter_kws(text_list):
    '''Filters out specific parts of speech and stop words from the list of potential keywords'''
    parsed_texts = nlp(' '.join(text_list))
    kw_filts = set([str(word) for word in parsed_texts
                if (word.pos_== ('NOUN' or 'ADJ' or 'VERB'))
                and word.lemma_ not in recipe_stopwords])

    if len(kw_filts) != len(set(kw_filts)):
        print("Duplicate words found in kw_filts. Removing duplicates.")
        kw_filts_list = list(set(kw_filts))  # Remove duplicates
    else:
        kw_filts_list = list(kw_filts)

    return kw_filts_list, parsed_texts

def generate_adjacency(kw_filts, parsed_texts):
    '''Tabulates counts of neighbors in the neighborhood window for each unique word'''
    adjacency = pd.DataFrame(columns=kw_filts, index=kw_filts, data = 0)

    for i, word in enumerate(parsed_texts):
        if any([str(word) == item for item in kw_filts]):
            if str(word) in adjacency.index:
                # Word already in the DataFrame, increment the count
                adjacency.loc[str(word), :] += 1
            else:
                # Word not in the DataFrame, add it with count 1
                end = min(len(parsed_texts), i + 5)  # Window of four words
                nextwords = parsed_texts[i + 1:end]
                inset = [str(x) in kw_filts for x in nextwords]
                neighbors = [str(nextwords[i]) for i in range(len(nextwords)) if inset[i]]
                if neighbors:
                    adjacency.loc[str(word), neighbors] += 1

    return adjacency

def run_textrank(adjacency, kw_filts):
    G = nx.DiGraph()

    for word, neighbors in adjacency.items():
        for neighbor, weight in neighbors.items():
            G.add_edge(word, neighbor, weight=weight)

    ranks = nx.pagerank(G, alpha=0.85, tol=1e-8)
    ranked = sorted(((ranks[word], word) for word in kw_filts if word in ranks), reverse=True)

    return ranked

def generate_wordranks(adjacency):
    '''Runs TextRank on adjacency table'''
    nx_words = nx.from_numpy_array(adjacency.values)
    ranks=nx.pagerank(nx_words, alpha=.85, tol=.00000001)

    return ranks

def generate_tag_list(ranks, kw_filts):
    '''Uses TextRank ranks to return actual key words for each topic in rank order'''
    rank_values = [i for i in ranks.values()]
    ranked = pd.DataFrame(zip(rank_values, list(kw_filts))).sort_values(by=0,axis=0,ascending=False)
    kw_list = ranked.iloc[:N_top_words,1].to_list()
    return kw_list

# Master Function utilizing all above functions
def generate_tags(topic_document_scores, root_text_data):
    text_4kwsummary = topic_docs_4kwsummary(topic_document_scores, root_text_data)
    kw_filts, parsed_texts = generate_filter_kws(text_4kwsummary)
    adjacency = generate_adjacency(kw_filts, parsed_texts)
    ranks = generate_wordranks(adjacency)
    kw_list = generate_tag_list(ranks, kw_filts)
    return kw_list

def generate_kw_index(topic_document_scores):
    kw_index = pd.Series(topic_document_scores).sort_values(ascending = False)[:N_docs_categorized].index
    return kw_index

# Set All Recommendation Model Parameters
N_topics = 50             # Number of Topics to Extract from corpora
N_top_docs = 200          # Number of top documents within each topic to extract keywords
N_top_words = 25          # Number of keywords to extract from each topic
N_docs_categorized = 2000 # Number of top documents within each topic to tag
N_neighbor_window = 4     # Length of word-radius that defines the neighborhood for
                          # each word in the TextRank adjacency table

# Query Similarity Weights
w_title = 0.2
w_text = 0.3
w_categories = 0.5
w_array = np.array([w_title, w_text, w_categories])

# Recipe Stopwords: for any high volume food recipe terminology that doesn't contribute
# to the searchability of a recipe. This list must be manually created.
recipe_stopwords = ['cup','cups','ingredient','ingredients','teaspoon','teaspoons','tablespoon',
                   'tablespoons','C','F']

csv_save_directory = 'C:/Users/yewji/FYP_20297501/server/recipe_recommendation/tf_idf/csv'
recipes = pd.read_csv("C:/Users/yewji/FYP_20297501/server/recipe_recommendation/tf_idf/csv/recipes.csv")
recipes = recipes.drop('image', axis=1)

# Indexing rows with columns that only contain numbers or punctuation
nc_ingred_index = index_categories(recipes, 'ingredients')
nc_title_index = index_categories(recipes, 'title')
nc_instr_index = index_categories(recipes, 'directions')

# Drop rows with only punctuation or numbers
inds_to_drop = set(reduce(add, [nc_ingred_index, nc_title_index, nc_instr_index]))
recipes = drop_rows_by_indices(recipes, [nc_ingred_index, nc_title_index, nc_instr_index])

# Process recipes
recipes = process_recipes(recipes)
print(recipes[['ingredient_text', 'ingredient_count', 'directions', 'directions_count']].head())
file_path = os.path.join(csv_save_directory, 'processed_recipes.csv')

# Combine text data
all_text = recipes['title'] + ' ' + recipes['ingredient_text'] + ' ' + recipes['directions']
print(all_text[0])

# Cleaning Text
cleaned_text = clean_text(all_text)

# Tokenize text
print("Tokenizing Text")
tokenized_text = [text_tokenizer_mp(text) for text in cleaned_text]
print(tokenized_text[0])

# Save the tokenized_text variable as a csv in order to return to it;
tokenized_text_df = pd.DataFrame({'text': tokenized_text})
# Save the DataFrame to a specific directory
file_path = os.path.join(csv_save_directory, 'tokenized_text.csv')
tokenized_text_df.to_csv(file_path, index_label='index')

# Creating Word Embeddings
text_tfidf, tfidf_words = generate_word_embeddings(tokenized_text)

print("(Num Topics/ Num Terms):", text_tfidf.shape) #(num of topics / recipes, number of unique terms / words)

# Topic Modeling
# LDA
lda = LDA(n_components=50, n_jobs=-1, max_iter=100)
text_lda = lda.fit_transform(text_tfidf)

# NNMF
nmf = NMF(init='nndsvdar', l1_ratio=0.0, max_iter=100, n_components=50, solver='cd')
text_nmf = nmf.fit_transform(text_tfidf)

# Exploring Topics by Document
# variable dependencies:
text_series = pd.Series(all_text)

# Exploring Topics by Document
print("LDA\n")
docs_by_tops(text_lda, (0, 3), (0, 3))
print("NMF\n")
docs_by_tops(text_nmf, (0, 3), (0, 3))

word_series = pd.Series(tfidf_words)

# Keywords using LDA
print("Keywords using LDA\n")
words_by_tops(text_tfidf, text_lda, (0,3), 10)
# Keywords using NMF
print("\nKeywords using NMF\n")
words_by_tops(text_tfidf, text_nmf, (0,3), 10)

# Renaming Data Dependencies
topic_transformed_matrix = text_nmf
root_text_data = cleaned_text

recipes['tag_list'] = [[] for i in repeat(None, recipes.shape[0])]

# Generating Tags and distributing to relevant documents
for i in range(topic_transformed_matrix.shape[1]):
    scores = topic_transformed_matrix[:, i]
    topic_kws = generate_tags(scores, root_text_data)
    kw_index_4df = generate_kw_index(scores)

    # Use append to add topic_kws to each list in 'tag_list'
    recipes.loc[kw_index_4df, 'tag_list'].apply(lambda x: x.extend(topic_kws) if isinstance(x, list) else topic_kws)

    if i % 10 == 0:
        print('Topic #{} Checkpoint'.format(i))
print('done!')

# Saving the precious dataframe so that I never have to calculate that again.
csv_save_path = os.path.join(csv_save_directory, 'tagged_recipes_df.csv')
recipes.to_csv(csv_save_path)

scores = topic_transformed_matrix[:,1]
topic_kws = generate_tags(scores, root_text_data)
kw_index_4df = generate_kw_index(scores)
recipes.loc[kw_index_4df, 'tag_list'].apply(lambda x: x.extend(topic_kws) if isinstance(x, list) else topic_kws)

# Concatenating lists of tags into a string a collective of tags for each documents
recipes['tags_string'] = [' '.join(tags) for tags in recipes['tag_list']]

# Saving the precious dataframe so that I never have to calculate that again.
csv_save_path = os.path.join(csv_save_directory, 'tagged_recipes_df.csv')
recipes.to_csv(csv_save_path)

print("New tagged_recipes_df.csv made")