import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from functools import reduce
import spacy

nlp = spacy.load("en_core_web_sm")

recipes = pd.read_csv('server/recipe_recommendation/tf_idf/csv/tagged_recipes_df.csv')

tokenized_text_path = 'server/recipe_recommendation/tf_idf/csv/tokenized_text.csv'
tokenized_text = pd.read_csv(tokenized_text_path)

# TF-IDF vectorizer instance
vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 1))

# Fit and transform the combined title and text data
text_tfidf = vectorizer.fit_transform(tokenized_text['text'].values.astype('U'))
title_tfidf = vectorizer.transform(recipes['title'].values.astype('U'))
tags_tfidf = vectorizer.transform(recipes['tags_string'].values.astype('U'))

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
    #final_scores = title_tfidf.multiply(query_vector * w_title) + text_tfidf.multiply(query_vector * w_text) + tags_tfidf.multiply(query_vector * w_categories)
    return final_scores

def print_recipes(index, query, recipe_range):
    '''Prints recipes according to query similary ranks'''
    print('Search Query: {}\n'.format(query))
    for i, index in enumerate(index, recipe_range[0]):
        print('Recipe Rank: {}\t'.format(i+1),recipes.loc[index, 'title'],'\n')
        print('Ingredients:\n{}\n '.format(recipes.loc[index, 'ingredient_text']))
        print('Directions:\n{}\n'.format(recipes.loc[index, 'directions']))

def Search_Recipes(query, query_ranked=False, recipe_range=(0, 3)):
    '''Master Recipe Search Function'''
    if query_ranked:  # Order of importance
        q_vector = ranked_query(query)
        print("Ranked:\n",q_vector)
    else:  # No order of importance, all equally important
        q_vector = vectorizer.transform([' '.join(query)])
        print("Not Ranked:\n",q_vector)
    recipe_scores = overall_scores(q_vector)
    sorted_index = pd.Series(recipe_scores.toarray().T[0]).sort_values(ascending=False)[recipe_range[0]:recipe_range[1]].index
    return print_recipes(sorted_index, query, recipe_range)

query = ['cinnamon', 'cream', 'banana']
Search_Recipes(query, query_ranked=True, recipe_range=(0, 3))
