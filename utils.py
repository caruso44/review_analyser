import csv
from string import punctuation, digits
import numpy as np

def read_file(file):
    tsv_file = open(file)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    train_text = []
    train_label = []
    for row in read_tsv:
        train_text.append(row[3])
        train_label.append(row[0])
    return train_text, train_label


def extract_words(input_string):
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()

def bag_of_words(texts):
    stopwords = set()
    f = open("Dataset/stopwords.txt")
    for x in f:
        aux = x.replace(x[len(x) - 1], "")
        stopwords.add(aux)
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary:
                if word not in stopwords:
                    dictionary[word] = len(dictionary)
    return dictionary

def extract_bow_feature_vectors(reviews, dictionary):
    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = 1
    return feature_matrix