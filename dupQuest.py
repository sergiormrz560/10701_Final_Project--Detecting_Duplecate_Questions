import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
#from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import nltk
from sklearn.decomposition import PCA
from matplotlib import pyplot
from numpy import linalg as LA


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
def getInputs(questions1, questions2, vectors, padding, k):
    vecInputs1, vecInputs2 = [], []
    for i in range(len(questions1)):
        inp1, inp2 = [], []

        #add padding at beginning of sentence
        for j in range(int((k-1)/2)):
            inp1.append(padding)
            inp2.append(padding)

        #add vector for each word in sentence
        [quest1, quest2] = tokenizeSentences([questions1[i], questions2[i]])
        for j in range(len(quest1)):
            inp1.append(vectors[quest1[j]])
        for j in range(len(quest2)):
            inp2.append(vectors[quest2[j]])

        #add padding at end of sentence
        for j in range(int((k-1)/2)):
            inp1.append(padding)
            inp2.append(padding)
        
        vecInputs1.append(inp1)
        vecInputs2.append(inp2)
    return vecInputs1, vecInputs2
        

#get the values of Z
def getZ(input1, input2, k):
    Z1, Z2 = [], []        
    for i in range(int(len(input1) - (k-1))):
        tempZ = np.array([])
        for j in range(k):
            tempZ = np.concatenate((tempZ, input1[i+j]))
        Z1.append(tempZ)
    for i in range(int(len(input2) - (k-1))):
        tempZ = np.array([])
        for j in range(k):
            tempZ = np.concatenate((tempZ, input2[i+j]))
        Z2.append(tempZ)
    return Z1, Z2


def f(x):
    return np.tanh(x)


def f_prime(y):
    return 1.0 - y**2


#calculate cosine similarity
def cosine_sim(vec1, vec2):
    return np.dot(vec1, vec2)/(LA.norm(vec1)*LA.norm(vec2))


#perform the forward pass
def feedForward(inputs1, inputs2, labels, d, k, clu, numSamples):
    W = np.random.rand(clu, d*k)
    b = np.random.rand(clu)
    rq1, rq2 = np.zeros(clu), np.zeros(clu)
    outputs = []
    for i in range(numSamples):
        Z1, Z2 = getZ(inputs1[i], inputs2[i], k)
        sum1, sum2 = 0, 0
        for j in range(len(Z1)):
            hiddenUnits1 = f(np.dot(W, Z1[j]) + b)
            sum1 = sum1 + hiddenUnits1
        for j in range(len(Z2)):
            hiddenUnits2 = f(np.dot(W, Z2[j]) + b)
            sum2 = sum2 + hiddenUnits2
            
        for j in range(len(rq1)):
            rq1[j] = f(sum1[j])
        for j in range(len(rq2)):
            rq2[j] = f(sum2[j])

        outputs.append(cosine_sim(rq1, rq2))
    return outputs

    

if __name__ == '__main__':
    df = pd.read_csv("questions.csv")
    questions1 = df['question1'][:200]
    questions2 = df['question2'][:200]
    questionPairs = np.array([questions1, questions2])
    questions = flattenData(questionPairs)
    tokenized_Questions = tokenizeSentences(questions)

    #k must be odd 
    k, clu, d = 5, 75, 100
    vectors = modelWord2Vec(tokenized_Questions, d)
    padding = vectors["the"]

    #get the questions into their respective vectors with necessary padding
    vecInputs1, vecInputs2 = getInputs(questions1, questions2, vectors, padding, k)
    labels = df['is_duplicate']

    outputs = feedForward(vecInputs1, vecInputs2, labels, d, k, clu, len(questions1))
    
    print(outputs)


