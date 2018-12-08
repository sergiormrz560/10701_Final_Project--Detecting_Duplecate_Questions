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
import re
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
import datetime

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
        if not(type(sentence) == str):
            sentence = "not sentence"
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
def getEmbeddings(questions1, questions2, vectors, padding, k):
    vecInputs1, vecInputs2 = [], []
    #Go through all questions and create vector representation of each
    for i in range(len(questions1)):
        #Vector representations of our questions
        inp1, inp2 = [], []

        #add padding at beginning of sentence
        #Now looking at individual question
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

'''
#perform the forward pass
def buildInputsToCNN(inputs1, inputs2, labels, d, k, clu, numSamples):
    for i in range(numSamples):
        Z1, Z2 = getZ(inputs1[i], inputs2[i], k)
'''

def buildAllZs(vecInputs1, vecInputs2, k):
    allZs = []
    for i in range(len(vecInputs1)):
        allZs.append(getZ(vecInputs1[i], vecInputs2[i], k))
    return allZs



def trainCNN(trainZs, trainLabels, testZs, testLabels, clu, k, d, maxEpoch, threshold):
    #Initialize the network itself, loss function, learning rate, and optimizer
    CNN = NeuralNet()
    criterion = nn.MSELoss()
    learningRate = 0.005
    optimizer = torch.optim.SGD(CNN.parameters(), lr=learningRate)


    #GPU = None
    #Send our network, criterion, optimizer, previousLoss, lossIncreaseCounter, and lr to the GPU
    if torch.cuda.is_available():
        GPU = torch.device("cuda")
        CNN = CNN.to(GPU)
        criterion = criterion.to(GPU)
        #trainZs = trainZs.to(GPU)
        trainLabels = trainLabels.to(GPU)
        #testZs = testZs.to(GPU)
        testLabels = testLabels.to(GPU)
        #optimizer = optimizer.to(GPU)
        #previousLoss = previousLoss.to(GPU)
        #lossIncreaseCounter = lossIncreaseCounter.to(GPU)
        #learningRate = learningRate.to(GPU)
        print("Using GPU")


    epochIndex = 0
    loss = 10
    #while(epochIndex < maxEpoch) and (loss > threshold):
    while(epochIndex < maxEpoch):
        #1 epoch of training
        for i in range(len(trainZs)):
            output = CNN(torch.from_numpy(np.array(trainZs[i][0])).float(), torch.from_numpy(np.array(trainZs[i][1])).float(), clu)
            #output = CNN(trainZs[i][0], trainZs[i][1], clu)
            loss = criterion(output, trainLabels[i])
            # clear gradients for next train
            optimizer.zero_grad()
            #Compute gradients and update weights
            loss.backward()
            optimizer.step()
            if(i%10000 == 0):
                print(datetime.datetime.now())

        #Calculate outputs on validation set
        netOutputs = torch.zeros(len(testZs))
        for i in range(len(testZs)):
            netOutputs[i] = CNN(torch.from_numpy(np.array(testZs[i][0])).float(), torch.from_numpy(np.array(testZs[i][1])).float(), clu)
            #netOutputs[i] = CNN(testZs[i][0], testZs[i][1], clu)

        #Calculate our accuracy on the validation set
        valAccuracy = valCalcAccuracy(netOutputs, testLabels)
        print(valAccuracy, "    ", datetime.datetime.now())
        epochIndex = epochIndex + 1
    return CNN

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


#Do heavy computations on the GPU if avaliable
'''if torch.cuda.is_available():
    batchData = batchData.to(GPU)
    batchLabels = batchLabels.to(GPU)
    output = neuralNetwork(batchData)
    loss = criterion(output, batchLabels)
    print(loss)
    # clear gradients for next train
    optimizer.zero_grad()
    #Compute gradients and update weights
    loss.backward()
    optimizer.step()
    if(loss > previousLoss):
        lossIncreaseCounter = lossIncreaseCounter + 1
        if(lossIncreaseCounter > 3):
            lr = lr / 2
            optimizer = torch.optim.SGD(neuralNetwork.parameters(), lr=learningRate)
'''

#No GPU avaliable
#else:
'''
output = neuralNetwork(batchData)
loss = criterion(output, batchLabels)
print(loss)
# clear gradients for next train
optimizer.zero_grad()
#Compute gradients and update weights
loss.backward()
optimizer.step()
if(loss > previousLoss):
    lossIncreaseCounter = lossIncreaseCounter + 1
    if(lossIncreaseCounter > 3):
        lr = lr / 2
        optimizer = torch.optim.Adam(neuralNetwork.parameters(), lr=learningRate)
'''


#-------------------------------Neural Network Layout---------------------------
#Class: NeuralNet
#Purpose: Builds the network along with the activation functions,
#         number of nodes per layer, and individual layers themselves
class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()
        # input: d-length of word embed, k-size of window
        # output: clu-sentence representation
        if torch.cuda.is_available():
            self.covolutionLayer = nn.Linear(d*k,  clu).cuda()
        else:
            self.covolutionLayer = nn.Linear(d*k, clu)


    def __call__(self, sentenceZns1, sentenceZns2, clu):
        return self.forward(sentenceZns1, sentenceZns2, clu)

    #sentenceZns-Matrix of zn's for every window around words in sentence
    def forward(self, sentenceZns1, sentenceZns2, clu):
        outputCNN1 = torch.zeros([len(sentenceZns1), clu])
        outputCNN2 = torch.zeros([len(sentenceZns2), clu])
        GPU = None
        if torch.cuda.is_available():
            GPU = torch.device("cuda")
            outputCNN1 = outputCNN1.to(GPU)
            outputCNN2 = outputCNN2.to(GPU)
            for i in range(len(sentenceZns1)):
                outputCNN1[i] = torch.tanh(self.covolutionLayer(sentenceZns1[i].float().to(GPU)))
            for i in range(len(sentenceZns2)):
                outputCNN2[i] = torch.tanh(self.covolutionLayer(sentenceZns2[i].float().to(GPU)))
        else:
            for i in range(len(sentenceZns1)):
                outputCNN1[i] = torch.tanh(self.covolutionLayer(sentenceZns1[i].float()))
            for i in range(len(sentenceZns2)):
                outputCNN2[i] = torch.tanh(self.covolutionLayer(sentenceZns2[i].float()))
        rq1 = torch.tanh(torch.sum(outputCNN1, dim=0))
        rq2 = torch.tanh(torch.sum(outputCNN2, dim=0))
        #print(rq1)
        cosSim = nn.CosineSimilarity(dim=0)
        output = cosSim(rq1, rq2)
        #print(output)
        return output
#-------------------------------------------------------------------------------


if __name__ == '__main__':
    #Read in the dataset
    df = pd.read_csv("dataset/questions.csv", encoding='ISO-8859-1')
    #df = pd.read_csv("dataset/questions.csv", encoding='utf-8')
    '''
    invalid_questions = []
    for i in range(len(df['question1'])):
        # questions need to contain a vowel (which should be part of a full word) to be valid
        if not re.search('[aeiouyAEIOUY]', df['question1'][i]) or not re.search('[aeiouyAEIOUY]', df['question2'][i]):
            # Need to subtract 'len(invalid_questions)' to adjust for the changing index values as questions are removed.
            invalid_questions.append(i-len(invalid_questions))
    print(invalid_quesions)
    '''

    # Many datapoints with garbage numbers
    df['question1'] = df['question1'].str.replace('[^a-zA-Z ]', '')
    df['question2'] = df['question2'].str.replace('[^a-zA-Z ]', '')
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)

    #will need to remove this
    df = df.head(50)

    '''
    #Put dataset into format to calc word embeddings
    questions1 = df['question1']
    questions2 = df['question2']
    questionPairs = np.array([questions1, questions2])
    questions = flattenData(questionPairs)
    tokenized_Questions = tokenizeSentences(questions)
    '''
    
    #calc word embeddings, d is the size of word embeddings
    #Hyper-parameters
    #
    # k is window size
    # clu is final size of sentence (sentence embedding)
    # d is dimensionality of word embedding
    k, clu, d = 3, 300, 200
    vectors = Word2Vec.load('word2vec.model')
    print(vectors)
    #vectors = modelWord2Vec(tokenized_Questions, d)
    #pcaVisulaization(model)

    #split dataset into train and test sets
    train, test = train_test_split(df, test_size=0.2)
    #reset the indexes of train and test sets
    train = train.reset_index()
    test = test.reset_index()


    padding = vectors["the"]

    #Put questions from train set into format to calculate z vectors
    questions1 = train['question1']
    questions2 = train['question2']
    #questionPairs = np.array([questions1, questions2])
    #questions = flattenData(questionPairs)
    #tokenized_Questions = tokenizeSentences(questions)
    #get the questions into their respective vectors with necessary padding
    vecInputs1, vecInputs2 = getEmbeddings(questions1, questions2, vectors, padding, k)
    #Get training set labels and Z's
    trainLabels = torch.from_numpy(train['is_duplicate'].values).float()
    trainZs = buildAllZs(vecInputs1, vecInputs2, k)

    #Put questions from test set into format to calculate z vectors
    questions1 = test['question1']
    questions2 = test['question2']
    #questionPairs = np.array([questions1, questions2])
    #questions = flattenData(questionPairs)
    #tokenized_Questions = tokenizeSentences(questions)
    #get the questions into their respective vectors with necessary padding
    vecInputs1, vecInputs2 = getEmbeddings(questions1, questions2, vectors, padding, k)
    #Get test set labels and Z's
    testLabels = torch.from_numpy(test['is_duplicate'].values).float()
    testZs = buildAllZs(vecInputs1, vecInputs2, k)

    maxEpoch, threshold = 100, 0.005
    CNN = trainCNN(trainZs, trainLabels, testZs, testLabels, clu, k, d, maxEpoch, threshold)
    #testCNN()

    #outputs = buildInputsToCNN(vecInputs1, vecInputs2, labels, d, k, clu, len(questions1))
