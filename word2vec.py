#Developer:
#Date:
#Class:
#Purpose:

import numpy as np
import csv
from gensim.models import Word2Vec
import nltk
from sklearn.decomposition import PCA
from matplotlib import pyplot

#Should work but getting a memory error on my local machine loading this in
def readTrainData():
    #BELOW IS THE ACTUAL LINE FOR READING IN OUR DATASET
    #reader = csv.reader(open("dataset/questions.csv", "rt"), delimiter=",")
    #THIS LINE IS USED FOR A SMALL SUBPORTION OF THE DATASET ONLY
    reader = csv.reader(open("dataset/smallTestSet.csv", "rt", encoding='utf-8', errors='ignore'), delimiter=",")
    trainData = list(reader)
    trainData = np.array(trainData).astype("object")
    return trainData

def stripData(data):
    #This removes the first line and first 3 columns
    return data[1:,3:5]

def flattenData(sentencePairs):
    sentences = sentencePairs.flatten()
    sentences = sentences.tolist()
    return sentences

def tokenizeSentences(sentences):
    tokenizedSentences = []
    for sentence in sentences:
        tokenized_text = nltk.word_tokenize(sentence)
        tokenizedSentences.append(tokenized_text)
    return tokenizedSentences

def modelWord2Vec(sentences):
    model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=1, sg=0)
    return model

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


if __name__ == '__main__':
    trainData = readTrainData()
    sentencePairs = stripData(trainData)
    sentences = flattenData(sentencePairs)
    tokenizedSentences = tokenizeSentences(sentences)
    model = modelWord2Vec(tokenizedSentences)
    pcaVisulaization(model)

    #CAN BE USED TO PRINT ALL THE WORDS TRAINED W/ WORD2VEC
    #words = list(model.wv.vocab)
    #print(words)

    #CAN BE USED TO PRINT THE WORD EMBEDDING OF 'AMERICA'
    #print(model['America'])
    #print(tokenizedSentences)

    #TO SAVE OUR LEARNED WORD EMBEDDINGS IN BINARY
    #model.wv.save_word2vec_format('model.bin')

    #TO SAVE OUR WORD EMBEDDINGS IN ASCII
    #model.wv.save_word2vec_format('model.txt', binary=False)

    #TO LOAD OUR SAVED MODEL
    #model = Word2Vec.load('model.bin')
