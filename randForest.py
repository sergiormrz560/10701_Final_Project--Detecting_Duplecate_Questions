#Libraries used
#import math
import numpy as np
import torch
import torch.nn as nn
#import torch.nn.functional as functional
import pandas as pd
import nltk
from nltk.corpus import stopwords
#from nltk.stem import SnowballStemmer
#import re
#from gensim import utils
#from gensim.models.doc2vec import LabeledSentence
#from gensim.models import Doc2Vec
#from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
#from numpy import linalg as LA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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


#get the vector form of the input questions
def getEmbeddings(questions1, questions2, vectors):
    vecInputs1, vecInputs2 = [], []
    #Go through all questions and create vector representation of each
    for i in range(len(questions1)):
        #Vector representations of our questions
        inp1, inp2 = [], []

        #add vector for each word in sentence
        [quest1, quest2] = tokenizeSentences([questions1[i], questions2[i]])
        for j in range(len(quest1)):
            inp1.append(vectors[quest1[j]])
        for j in range(len(quest2)):
            inp2.append(vectors[quest2[j]])

        vecInputs1.append(inp1)
        vecInputs2.append(inp2)

    return vecInputs1, vecInputs2


def valCalcAccuracy(netOutputs, labels):
    correct = 0
    index = 0
    while(index < len(labels)):
        if(labels[index] == 1):
            if(netOutputs[index] > 0.5):
                correct = correct + 1
        #must be 0
        else:
            if(netOutputs[index] < 0.5):
                correct = correct + 1
        index = index + 1
    return correct / len(netOutputs)

def calcAvgEmbedding(questions, d):
    questionEmbeddings = np.zeros([len(questions), d])
    index = 0
    while(index < len(questions)):
        sentenceEmbeddings = np.asarray(questions[index])
        questionEmbeddings[index] = np.average(sentenceEmbeddings, axis=0)
        index = index + 1
    return questionEmbeddings

def calcEmbedSentencePair(question1PerSentenceAvg, question2PerSentenceAvg, d):
    questionEmbeddings = np.zeros([len(question1PerSentenceAvg), d*2])
    index = 0
    while(index < len(question1PerSentenceAvg)):
        questionEmbeddings[index][0:100] = question1PerSentenceAvg[index]
        questionEmbeddings[index][100:200] = question2PerSentenceAvg[index]
        index = index + 1
    return questionEmbeddings

def valCalcAccuracy(outputs, labels):
    correct = 0
    index = 0
    while(index < len(labels)):
        if(labels[index] == 1):
            if(outputs[index] == 1):
                correct = correct + 1
        #must be 0
        else:
            if(outputs[index] == 0):
                correct = correct + 1
        index = index + 1
    return correct / len(outputs)


if __name__ == '__main__':
    #Read in the dataset
    df = pd.read_csv("dataset/questions.csv", encoding='ISO-8859-1')
    #will need to remove this
    df = df.head(5)

    #Put dataset into format to calc word embeddings
    questions1 = df['question1']
    questions2 = df['question2']
    questionPairs = np.array([questions1, questions2])
    questions = flattenData(questionPairs)
    tokenized_Questions = tokenizeSentences(questions)

    #calc word embeddings, d is the size of word embeddings
    d = 100
    vectors = modelWord2Vec(tokenized_Questions, d)
    #pcaVisulaization(model)

    #split dataset into train and test sets
    train, test = train_test_split(df, test_size=0.2)
    #reset the indexes of train and test sets
    train = train.reset_index()
    test = test.reset_index()

    #Put questions from train set into format to calculate z vectors
    questions1 = train['question1']
    questions2 = train['question2']
    questionPairs = np.array([questions1, questions2])
    questions = flattenData(questionPairs)
    tokenized_Questions = tokenizeSentences(questions)
    #get the questions into their respective vectors
    embeddings1, embeddings2 = getEmbeddings(questions1, questions2, vectors)
    #Calculate the average word embedding across a sentence
    question1PerSentenceAvg = calcAvgEmbedding(embeddings1, d)
    question2PerSentenceAvg = calcAvgEmbedding(embeddings2, d)
    questionCombinedEmbed = calcEmbedSentencePair(question1PerSentenceAvg, question2PerSentenceAvg, d)
    trainLabels = train['is_duplicate'].values

    #Initialize our random forest classifier and train
    randForest = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=10,  random_state=0)
    randForest.fit(questionCombinedEmbed, trainLabels)


    #Put questions from test set into format to calculate z vectors
    questions1 = test['question1']
    questions2 = test['question2']
    questionPairs = np.array([questions1, questions2])
    questions = flattenData(questionPairs)
    tokenized_Questions = tokenizeSentences(questions)
    #get the questions into their respective vectors with necessary padding
    embeddings1, embeddings2 = getEmbeddings(questions1, questions2, vectors)
    question1PerSentenceAvgTest = calcAvgEmbedding(embeddings1, d)
    question2PerSentenceAvgTest = calcAvgEmbedding(embeddings2, d)
    questionCombinedEmbedTest = calcEmbedSentencePair(question1PerSentenceAvg, question2PerSentenceAvg, d)
    testLabels = test['is_duplicate'].values

    outputs = randForest.predict(questionCombinedEmbedTest)
    valAccuracy = valCalcAccuracy(outputs, testLabels)
    print(valAccuracy)
