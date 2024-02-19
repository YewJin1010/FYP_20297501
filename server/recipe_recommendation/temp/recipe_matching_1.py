import pandas as pd
# Read files
recipes = pd.read_csv("/content/drive/MyDrive/FYP_Datasets/recipes.csv")
recipes = recipes.drop('image', axis=1)

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

# Check information about the dataset
print("Recipes Info")
print("\nDataset Information:")
print(recipes.info())

print("\nDataset Shape:")
print(recipes.shape)

# Count missing values by category
print("\nMissing Values Count:")
print(recipes.isna().sum())

print("\nData Types:")
print(recipes.dtypes)

# JohnVillanueva recommendation system
def index_categories(data, column):
    return [index for i, index in zip(data[column], data.index) if all(j.isdigit() or j in string.punctuation for j in i)]

def drop_rows_by_indices(data, indices):
    return data.drop(index=set(reduce(add, indices))).reset_index(drop=True)

# Indexing rows with columns that only contain numbers or punctuation
nc_ingred_index = index_categories(recipes, 'ingredients')
nc_title_index = index_categories(recipes, 'title')
nc_instr_index = index_categories(recipes, 'directions')

# Drop rows with only punctuation or numbers
inds_to_drop = set(reduce(add, [nc_ingred_index, nc_title_index, nc_instr_index]))
print(len(inds_to_drop))
recipes = drop_rows_by_indices(recipes, [nc_ingred_index, nc_title_index, nc_instr_index])
print(recipes.shape)

# Drop recipe instructions with less than 20 characters
empty_instr_ind = [index for i, index in zip(recipes['directions'], recipes.index) if len(i) < 20]
recipes = recipes.drop(index=empty_instr_ind).reset_index(drop=True)
print(recipes.shape)

# Checking for low ingredient recipes.
low_ingr_index = [index for i, index in zip(recipes['ingredients'], recipes.index) if i[0] == np.nan]
print(len(low_ingr_index))
print(recipes.loc[low_ingr_index, 'ingredients'])

# Convert string representations of lists to actual lists for all recipes
recipes['ingredients'] = recipes['ingredients'].apply(ast.literal_eval)

# Extracting ingredients from their lists and formatting as single strings
recipes['ingredient_text'] = ['; '.join(ingredients) for ingredients in recipes['ingredients']]
print(recipes['ingredient_text'].head())

# Counting the number of ingredients used in each recipe
recipes['ingredient_count'] = [len(ingredients) for ingredients in recipes['ingredients']]
print(recipes['ingredient_count'].head())

# Convert string representations of lists to actual lists for all recipes
recipes['directions'] = recipes['directions'].apply(ast.literal_eval)
recipes['directions'] = [' '.join(instruction[1:]).strip('[]') if index > 0 else ' '.join(instruction).strip('[]') for index, instruction in enumerate(recipes['directions'])]

print(recipes['directions'].head())

recipes.head(1)

all_text = recipes['title'] + ' ' + recipes['ingredient_text'] + ' ' + recipes['directions']
all_text[0]

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
    nlp = spacy.load("en_core_web_sm")
    tokenized_documents = [
        ' '.join([token.lemma_ for token in nlp(doc) if not token.is_stop])
        for doc in documents
    ]
    return tokenized_documents

# Tokenizing Using Spacy and run in parallel
def text_tokenizer_mp(doc):
    nlp = spacy.load("en_core_web_sm")
    return ' '.join([token.lemma_ for token in nlp(doc) if not token.is_stop])

# Cleaning Text
cleaned_text = clean_text(all_text)

# Cleaned Text
cleaned_text[2]

# Number of processors
num_processors = mp.cpu_count()
print("Number of processors: ", num_processors)

# Parallelizing tokenizing process
with mp.Pool(num_processors) as pool:
    tokenized_text = pool.map(text_tokenizer_mp, cleaned_text)

# Save the tokenized_text variable as a csv in order to return to it;
tokenized_text_df = pd.DataFrame({'text': tokenized_text})

# Save the DataFrame
tokenized_text_df.to_csv('tokenized_text.csv', index_label='index')

tokenized_text[0]

# Creating Word Embeddings
vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 1))
text_tfidf = vectorizer.fit_transform(tokenized_text)
tfidf_words = vectorizer.get_feature_names_out()

print(text_tfidf.shape) #(num of topics / recipes, number of unique terms / words)

# Topic Modeling
# LDA
lda = LDA(n_components=50, n_jobs=-1, max_iter=100)
text_lda = lda.fit_transform(text_tfidf)
text_lda.shape

# NNMF
nmf = NMF(init='nndsvdar', l1_ratio=0.0, max_iter=100, n_components=50, solver='cd')
text_nmf = nmf.fit_transform(text_tfidf)
text_nmf.shape

# Exploring Topics by Document
# variable dependencies:
text_series = pd.Series(all_text)

def docs_by_tops(top_mat, topic_range = (0,0), doc_range = (0,2)):
    for i in range(topic_range[0], topic_range[1]):
        topic_scores = pd.Series(top_mat[:,i])
        doc_index = topic_scores.sort_values(ascending = False)[doc_range[0]:doc_range[1]].index
        for j, index in enumerate(doc_index, doc_range[0]):
            print('Topic #{}'.format(i),
                  '\nDocument #{}'.format(j),
                  '\nTopic Score: {}\n\n'.format(topic_scores[index]),
                  text_series[index], '\n\n')

print("LDA\n")
docs_by_tops(text_lda, (0, 3), (0, 3))
print("NMF\n")
docs_by_tops(text_nmf, (0, 3), (0, 3))

# Exploring Topics by words
print(text_nmf.shape) # (num doc/recipes, num unique words/terms)
print(text_tfidf.T.shape) # (num unique terms/words, recipes) after transposing

# Function for best topic words using cosine similarity
# Variable Dependency:
word_series = pd.Series(tfidf_words)

def words_by_tops(tfidf_mat, top_mat, topic_range=(0,0), n_words=10):
    topic_word_scores = tfidf_mat.T * top_mat
    for i in range(topic_range[0],topic_range[1]):
        word_scores = pd.Series(topic_word_scores[:,i])
        word_index = word_scores.sort_values(ascending = False)[:n_words].index
        print('\nTopic #{}'.format(i))
        for index in word_index:
            print(word_series[index],'\t\t', word_scores[index])

# Keywords using LDA
print("Keywords using LDA\n")
words_by_tops(text_tfidf, text_lda, (0,3), 10)
# Keywords using NMF
print("\nKeywords using NMF\n")
words_by_tops(text_tfidf, text_nmf, (0,3), 10)

# Keyword Extraction using TextRank
def generate_filter_kws(text_list, recipe_stopwords):
    nlp = spacy.load("en_core_web_sm")
    parsed_texts = nlp(' '.join(text_list))
    kw_filts = set([str(word) for word in parsed_texts
                    if (word.pos_ in ('NOUN', 'ADJ', 'VERB'))
                    and str(word) not in recipe_stopwords])

    if len(kw_filts) != len(set(kw_filts)):
        print("Duplicate words found in kw_filts. Removing duplicates.")
        kw_filts_list = list(set(kw_filts))
    else:
        kw_filts_list = list(kw_filts)

    return kw_filts_list, parsed_texts

def generate_adjacency(kw_filts, parsed_texts):
    adjacency = pd.DataFrame(columns=kw_filts, index=kw_filts, data=0)

    for i, word in enumerate(parsed_texts):
        if any([str(word) == item for item in kw_filts]):
            if str(word) in adjacency.index:
                adjacency.loc[str(word), :] += 1
            else:
                end = min(len(parsed_texts), i + 5)
                nextwords = parsed_texts[i + 1:end]
                inset = [str(x) in kw_filts for x in nextwords]
                neighbors = [str(nextwords[i]) for i in range(len(nextwords)) if inset[i]]
                if neighbors:
                    adjacency.loc[str(word), neighbors] += 1

    return adjacency

def run_textrank(adjacency):
    G = nx.DiGraph()

    for word, neighbors in adjacency.items():
        for neighbor, weight in neighbors.items():
            G.add_edge(word, neighbor, weight=weight)

    ranks = nx.pagerank(G, alpha=0.85, tol=1e-8)
    ranked = sorted(((ranks[word], word) for word in kw_filts if word in ranks), reverse=True)

    return ranked

# Load a sample of documents for keyword extraction
text_index = pd.Series(text_nmf[:, 1]).sort_values(ascending=False)[:100].index
text_4summary = pd.Series(cleaned_text)[text_index]

# Manually create a list of recipe stopwords
recipe_stopwords = ['cup', 'cups', 'ingredient', 'ingredients', 'teaspoon', 'tablespoon', 'oven']

# Generate keyword filters and adjacency matrix
kw_filts, parsed_texts = generate_filter_kws(text_4summary, recipe_stopwords)
adjacency = generate_adjacency(kw_filts, parsed_texts)

# Run TextRank
ranked_keywords = run_textrank(adjacency)

ranked_keywords[:25]

adjacency.shape

# checking to see there are actual values loaded in the adjacency df
import scipy
scipy.sparse.csr_matrix(adjacency.copy().values)

pd.Series(list(kw_filts)).nunique()

print("Length of text_4summary:", len(text_4summary))
print("Available indices in text_4summary:", text_4summary.index)

text_4summary.iloc[90]

import matplotlib.pyplot as plt
# text_lda
# text_nmf
# ranked

# LDA Topic documents for topics 0-2
plt.figure(figsize=(15,4))
for i in range(3):
    series = pd.Series(text_lda[:,i])
    plt.subplot(1,3,i+1)
    plt.hist(series[series > 0.05])
    plt.title('LDA Topic #{} Doc Score Dist (>0.05)'.format(i+1))
plt.show()

# NNMF Topic documents for topics 0-2
plt.figure(figsize=(15,4))
for i in range(3):
    series = pd.Series(text_nmf[:,i])
    plt.subplot(1,3,i+1)
    plt.hist(series[series > 0.004])
    plt.title('NNMF Topic #{} Document Score Dist (>0.004)'.format(i+1))
    plt.xlabel('Document Topic Score')
#plt.savefig('DocsByTop_Score_Distributions.png', transparent = True)
plt.show()

# LDA Topic document scores for topics 0-2
plt.figure(figsize=(15,4))
for i in range(3):
    series = pd.Series(text_lda[:,i]).copy().sort_values(ascending = False).reset_index(drop = True)
    plt.subplot(1,3,i+1)
    plt.plot(series[:1000])
    plt.title('LDA Topic #{} Ordered Score Plot'.format(i+1))
plt.show()

# NMF Topic document scores for topics 0-2
plt.figure(figsize=(15,4))
for i in range(3):
    series = pd.Series(text_nmf[:,i]).copy().sort_values(ascending = False).reset_index(drop = True)
    plt.subplot(1,3,i+1)
    plt.plot(series[:1000])
    plt.title('NMF Topic #{} Ordered Score Plot'.format(i+1))
    plt.xlabel('Document Rank')
    plt.ylabel('Document Topic Score')
#plt.savefig('DocsByTop_Score_Elbows.png', transparent = True)
plt.show()

# Putting it all together
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

# Renaming Data Dependencies
topic_transformed_matrix = text_nmf
root_text_data = cleaned_text

# Generating tags (keywords/categories) and assigning to corresponding documents
from itertools import repeat

recipes['tag_list'] = [[] for i in repeat(None, recipes.shape[0])]
nlp = spacy.load("en_core_web_sm")

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

def generate_wordranks(adjacency):
    '''Runs TextRank on adjacency table'''
    nx_words = nx.from_numpy_array(adjacency.values)
    ranks=nx.pagerank(nx_words, alpha=.85, tol=.00000001)

    return ranks

def generate_tag_list(ranks):
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
    kw_list = generate_tag_list(ranks)
    return kw_list

def generate_kw_index(topic_document_scores):
    kw_index = pd.Series(topic_document_scores).sort_values(ascending = False)[:N_docs_categorized].index
    return kw_index

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
recipes.to_csv('tagged_recipes_df.csv')

scores = topic_transformed_matrix[:,1]
topic_kws = generate_tags(scores, root_text_data)
kw_index_4df = generate_kw_index(scores)
recipes.loc[kw_index_4df, 'tag_list'].apply(lambda x: x.extend(topic_kws) if isinstance(x, list) else topic_kws)

recipes.loc[:5,'tag_list']

# Concatenating lists of tags into a string a collective of tags for each documents
recipes['tags'] = [' '.join(tags) for tags in recipes['tag_list']]

recipes.to_csv('updated_tag_recipes_df.csv')

recipes.loc[:5,'tags']

recipes.columns

tokenized_text_path = '/content/tokenized_text.csv'
tokenized_text = pd.read_csv(tokenized_text_path)

tokenized_text.head()

# Creating TF-IDF Matrices and recalling text dependencies
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

'''import text_tokenized.csv here'''
tokenized_text_path = '/content/tokenized_text.csv'
tokenized_text = pd.read_csv(tokenized_text_path)

# TF-IDF vectorizer instance
vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 1))

# Fit and transform the combined title and text data
text_tfidf = vectorizer.fit_transform(tokenized_text['text'].values.astype('U'))
title_tfidf = vectorizer.transform(recipes['title'].values.astype('U'))
tags_tfidf = vectorizer.transform(recipes['tags'].values.astype('U'))

# Query Similarity Weights
w_title = 0.2
w_text = 0.3
w_categories = 0.5

def qweight_array(query_length, qw_array = [1]):
    '''Returns descending weights for ranked query ingredients'''
    if query_length > 1:
        to_split = qw_array.pop()
        split = to_split/2
        qw_array.extend([split, split])
        return qweight_array(query_length - 1, qw_array)
    else:
        return np.array(qw_array)

def ranked_query(query):
    '''Called if query ingredients are ranked in order of importance.
    Weights and adds each ranked query ingredient vector.'''
    query = [[q] for q in query]      # place words in seperate documents
    q_vecs = [vectorizer.transform(q) for q in query]
    qw_array = qweight_array(len(query),[1])
    q_weighted_vecs = q_vecs * qw_array
    q_final_vector = reduce(np.add,q_weighted_vecs)
    return q_final_vector

def overall_scores(query_vector):
    '''Calculates Query Similarity Scores against recipe title, directions, and keywords.
    Then returns weighted averages of similarities for each recipe.'''
    final_scores = title_tfidf*query_vector.T*w_title
    final_scores += text_tfidf*query_vector.T*w_text
    final_scores += tags_tfidf*query_vector.T*w_categories
    return final_scores

def print_recipes(index, query, recipe_range):
    '''Prints recipes according to query similary ranks'''
    print('Search Query: {}\n'.format(query))
    for i, index in enumerate(index, recipe_range[0]):
        print('Recipe Rank: {}\t'.format(i+1),recipes.loc[index, 'title'],'\n')
        print('Ingredients:\n{}\n '.format(recipes.loc[index, 'ingredient_text']))
        print('directions:\n{}\n'.format(recipes.loc[index, 'directions']))

def Search_Recipes(query, query_ranked=False, recipe_range=(0,3)):
    '''Master Recipe Search Function'''
    if query_ranked == True: # Order of importance
        q_vector = ranked_query(query)
    else: # No order of importance, all equally important
        q_vector = vectorizer.transform([' '.join(query)])
    recipe_scores = overall_scores(q_vector)
    sorted_index = pd.Series(recipe_scores.toarray().T[0]).sort_values(ascending = False)[recipe_range[0]:recipe_range[1]].index
    return print_recipes(sorted_index, query, recipe_range)

query = ['cinnamon', 'cream', 'banana']
Search_Recipes(query, query_ranked=True, recipe_range=(0,3))

# Cleaning to Prepare for Tokenizing
# Removing ADVERTISEMENT text from ingredients list
ingredients = []
for ing_list in recipes['ingredients']:
    clean_ings = [ing.replace('ADVERTISEMENT','').strip() for ing in ing_list]
    if '' in clean_ings:
        clean_ings.remove('')
    ingredients.append(clean_ings)
recipes['ingredients'] = ingredients

# DMW project food recommendation system
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# Convert lists to strings
recipes['ingredients'] = recipes['ingredients'].apply(lambda x: ' '.join(x))
# Print the first two ingredients' lists after conversion
print("First ingredient list after conversion:")
print(recipes['ingredients'].iloc[0])
print("\nSecond ingredient list after conversion:")
print(recipes['ingredients'].iloc[1])

# Clean up extra spaces and characters
recipes['ingredients'] = recipes['ingredients'].str.replace("'", '').str.strip()

# Print the cleaned ingredients
print("\nCleaned ingredients:")
print(recipes['ingredients'])

# Convert lists to strings
non_empty_recipes['ingredients'] = non_empty_recipes['ingredients'].apply(lambda x: ' '.join(x)).str.strip()

# Print the processed ingredients
print("\nProcessed ingredients:")
print(non_empty_recipes['ingredients'])

# Fit and transform the 'ingredients' column
csr_dataset = vectorizer.fit_transform(non_empty_recipes['ingredients'].fillna(''))

# Print the shape of the resulting sparse matrix
print("\nShape of the resulting sparse matrix:")
print(csr_dataset.shape)

# Assuming 'ingredients' is a list of strings in each row
vectorizer = TfidfVectorizer(stop_words='english')

# Filter out rows with empty 'ingredients'
non_empty_recipes = recipes[recipes['ingredients'].apply(lambda x: bool(x))]

# Fit and transform the 'ingredients' column
csr_dataset = vectorizer.fit_transform(non_empty_recipes['ingredients'].apply(lambda x: ' '.join(x)).str.strip().fillna(''))

# Print the shape of the resulting sparse matrix
print("\nShape of the resulting sparse matrix:")
print(csr_dataset.shape)

# Using Algorithm
model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model.fit(csr_dataset)

# Remove Sparsity
csr_dataset = csr_matrix(dataset['ingredients'].fillna('').apply(lambda x: ' '.join(x.split())).str.split().apply(lambda x: ' '.join(x)))



# pivot table
dataset = ratings.pivot_table(index='Food_ID', columns='User_ID',values='Rating')
dataset.fillna(0,inplace=True)

# Recommendation function
def food_recommendation(Food_Name):
  n=10
  FoodList = food[food['Name'].str.contains(Food_Name)]
  if len(FoodList):
    Foodi = FoodList.iloc[0]['Food_ID']
    Foodi = dataset[dataset['Food_ID'] == Foodi].index[0]
    distances, indices = model.kneighbors(csr_dataset[Foodi],n_neighbors = n+1)
    Food_indices = sorted(list(zip(indices.squeeze().tolist(),distance.squeeze().tolist())),key=lambda x: x[1])
    Recommendations = []
    for val in Food_indices:
      Foodi = dataset.iloc[val[0]['Food_ID']]
      i = food[food['Food_ID'] == Foodi].index
      Recommendations.append({'Name': food.iloc[i]['Name'].values[0], 'Distance': val[1]})
    df = pd.DataFrame(Recommendations, index=range(1, n+1))
    return df['Name']
  else:
    return "No Similar Foods"

# Remove Sparsity
csr_dataset = csr_matrix(dataset.values)
dataset.reset_index(inplace = True)

# Using Algorithm
model = NearestNeighbors(metric= 'cosine', algorithm = 'brute', n_neighbors=20, n_jobs=-1)
model.fit(csr_dataset)

food_recommendation('cashew nut cookies')

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

data = pd.read_csv('/content/drive/MyDrive/FYP_Datasets/recipes.csv')
data.head()

data.info()

dataset=data.copy()
columns=['id', 'title', 'ingredients']
dataset=dataset[columns]

