import re
import numpy as np
import pandas as pd

# Regex for model words in title
#regex_title = re.compile(r'[a-zA-Z0-9]*(?:[0-9]+[^0-9, ]+|[^0-9, ]+[0-9]+)[a-zA-Z0-9]*')
regex_title = re.compile(r'\b[a-zA-Z0-9]*(?:[0-9]+[^0-9,\s]+|[^0-9,\s]+[0-9]+)[a-zA-Z0-9]*\b')
# Regex for model words in key-value pairs
regex_value = re.compile(r'^\d+(\.\d+)?[a-zA-Z]+$|^\d+(\.\d+)?$')


def meets_two_types(word):
    has_letter = bool(re.search(r'[a-zA-Z]', word))
    has_digit = bool(re.search(r'\d', word))
    has_special = bool(re.search(r'[^a-zA-Z0-9]', word))

    types_count = sum([has_letter, has_digit, has_special])
    return types_count >= 2

def extract_model_words_title(text):
    if not isinstance(text, str):
        return set()

    mw_title = set()  #set, so unique mws are storwed

    matches = regex_title.finditer(text)
    for match in matches:
        word = match.group(0)
        word = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', word)
        if meets_two_types(word): #needs to have at least 2 out of 3 types
            mw_title.add(word)
    return mw_title



#function to extract the set of model words from all titles
def extract_mw_title_column(title_series):
    MW_titles = set()
    for title in title_series:
        MW_titles.update(extract_model_words_title(title))
    return MW_titles  # return set of unique model words of all titles

#function that extracts model words from 1 string value/feature value
def extract_model_words_value(value):
    #Extract model words from a value string
    if not isinstance(value, str):
        return set()
    mw_value = set()
    #Split string to tokens
    tokens = re.findall(r'[^,\s]+', value)
    for token in tokens:     #Check for all tokens if it is a model word
        # Check if the whole token matches the value regex
        if regex_value.fullmatch(token):
            #Extract only the numeric part
            num_part = re.match(r'\d+(\.\d+)?', token)
            mw_value.add(num_part.group())
    return mw_value

#function that extracts model words from 1 row/1 dictionary of featuresMap
def extract_model_words_value_dict(features_map):
    mw = set()
    if not isinstance(features_map, dict):
        return mw
    for val in features_map.values():
        mw.update(extract_model_words_value(val))
    return mw

#function that extracts model words on full column of featuresMap
def extract_mw_FM_column(features_series):
    MW_value = set()
    for features_map in features_series:
        MW_value.update(extract_model_words_value_dict(features_map))
    return MW_value



def binary_matrix_func(MW_title, MW_value, title_mw_list, value_mw_list):
    # extra cleanin, some words contained diagonal after inch which is not supposed to happen
    def clean_mw(mw: str):
        s = str(mw)
        s = re.sub(r"diagonal", "", s, flags=re.IGNORECASE)
        return s
    MW_title = {
        clean_mw(mw)
        for mw in MW_title
        if clean_mw(mw) != ""
    }
    MW_list = list(MW_title.union(MW_value))

    title_mw_list = [
        {
            clean_mw(mw)
            for mw in mw_list
            if clean_mw(mw) != ""
        }
        for mw_list in title_mw_list
    ]
    ##############################
    #Make the matrix
    num_products = len(title_mw_list) #number of products/number of elements in title_mw_list
    num_elements = len(MW_list) #total number of model_words in MW_list

    matrix = np.zeros((num_elements, num_products)) #matrix with zeros, with per row the model words and per column the products

    for p in range(num_products):
        title_mw = title_mw_list[p]
        value_mw = value_mw_list[p]

        for i, mw in enumerate(MW_list):

            #title attribute contains mw from MWtitle
            if mw in title_mw and mw in MW_title:
                matrix[i, p] = 1
                continue

            #value attribute contains mw from MWtitle
            if mw in value_mw and mw in MW_title:
                matrix[i, p] = 1
                continue

            #value attribute contains mw from MWvalue
            if mw in value_mw and mw in MW_value:
                matrix[i, p] = 1
                continue

    return matrix




