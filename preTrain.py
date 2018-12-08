#Libraries used
#import math
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot

def pcaVisulaization(model):
    #To retrieve all vectors from a trained model:
    X = model[model.wv.vocab]
    #Plot word vectors using PCA (uses the scikit-learn PCA Class and matplotlib):
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    pyplot.scatter(result[:, 0], result[:, 1])
    #Annotates graph
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()


def tokenizeSentences(sentences):
    tokenizedSentences = []
    for sentence in sentences:
        tokenized_text = nltk.word_tokenize(sentence)
        tokenizedSentences.append(tokenized_text)
    return tokenizedSentences


def flattenData(sentencePairs):
    sentences = sentencePairs.flatten()
    sentences = sentences.tolist()
    return sentences

def modelWord2Vec(sentences, d):
    model = Word2Vec(sentences, size=d, window=5, min_count=1, workers=1, sg=0)
    return model


if __name__ == '__main__':
    #Read in the dataset
    df = pd.read_csv("dataset/questions.csv", encoding='ISO-8859-1')

    #Remove all 'extra' characters
    df['question1'] = df['question1'].str.replace('[^a-zA-Z0-9 ]', '')
    df['question2'] = df['question2'].str.replace('[^a-zA-Z0-9 ]', '')
    #Make lower case
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)

    #will need to remove this
    df = df.head(50)

    #Put dataset into format to calc word embeddings
    questions1 = df['question1']
    questions2 = df['question2']
    questionPairs = np.array([questions1, questions2])
    questions = flattenData(questionPairs)
    tokenized_Questions = tokenizeSentences(questions)

    #calc word embeddings, d is the size of word embeddings
    d = 200
    vectors = modelWord2Vec(tokenized_Questions, d)
    #pcaVisulaization(model)
    vectors.save('word2vec.model')
