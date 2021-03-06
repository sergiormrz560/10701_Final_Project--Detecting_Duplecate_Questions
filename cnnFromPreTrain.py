#Libraries used
#import math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from matplotlib import pyplot
from sklearn.model_selection import train_test_split


def tokenizeSentences(sentences):
    tokenizedSentences = []
    for i in range(len(sentences)):
        if not(type(sentences[i]) == str):
            sentences[i] = "no sentence"
        tokenized_text = nltk.word_tokenize(sentences[i])
        tokenizedSentences.append(tokenized_text)
    return tokenizedSentences


def flattenData(sentencePairs):
    sentences = sentencePairs.flatten()
    sentences = sentences.tolist()
    return sentences


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

    epochIndex = 0
    loss = 10
    #while(epochIndex < maxEpoch) and (loss > threshold):
    while(epochIndex < maxEpoch):
        #1 epoch of training
        for i in range(len(trainZs)):
            #output = CNN(torch.from_numpy(trainZs[i][0]).float(), torch.from_numpy(trainZs[i][1]).float(), clu)
            output = CNN(trainZs[i][0], trainZs[i][1], clu)
            loss = criterion(output, trainLabels[i])
            # clear gradients for next train
            optimizer.zero_grad()
            #Compute gradients and update weights
            loss.backward()
            optimizer.step()

        #Calculate outputs on validation set
        netOutputs = torch.zeros(len(testZs))
        for i in range(len(testZs)):
            #netOutputs[i] = CNN(torch.from_numpy(testZs[i][0]).float(), torch.from_numpy(testZs[i][1]).float(), clu)
            netOutputs[i] = CNN(testZs[i][0], testZs[i][1], clu)

        #Calculate our accuracy on the validation set
        valAccuracy = valCalcAccuracy(netOutputs, testLabels)
        print(valAccuracy)
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


#-------------------------------Neural Network Layout---------------------------
#Class: NeuralNet
#Purpose: Builds the network along with the activation functions,
#         number of nodes per layer, and individual layers themselves
class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()
        # input: d-length of word embed, k-size of window
        # output: clu-sentence representation
        self.covolutionLayer = nn.Linear(d * k,  clu)


    def __call__(self, sentenceZns1, sentenceZns2, clu):
        return self.forward(sentenceZns1, sentenceZns2, clu)

    #sentenceZns-Matrix of zn's for every window around words in sentence
    def forward(self, sentenceZns1, sentenceZns2, clu):
        outputCNN1 = torch.zeros([len(sentenceZns1), clu])
        outputCNN2 = torch.zeros([len(sentenceZns2), clu])
        #GPU = None
        #if torch.cuda.is_available():
        #    GPU = torch.device("cuda")
        #    outputCNN1 = outputCNN1.to(GPU)
        #    outputCNN2 = outputCNN2.to(GPU)
        for i in range(len(sentenceZns1)):
            outputCNN1[i] = torch.tanh(self.covolutionLayer(torch.from_numpy(sentenceZns1[i]).float()))
        for i in range(len(sentenceZns2)):
            outputCNN2[i] = torch.tanh(self.covolutionLayer(torch.from_numpy(sentenceZns2[i]).float()))
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

    #Remove all 'extra' characters
    df['question1'] = df['question1'].str.replace('[^a-zA-Z0-9 ]', '')
    df['question2'] = df['question2'].str.replace('[^a-zA-Z0-9 ]', '')
    #Make lower case
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)

    #will need to remove this
    df = df.head(50)

    #Load in pre-trained word embeddings
    vectors = Word2Vec.load('word2vec.model')
    #print(vectors)


    #Hyper-parameters
    k, clu, d = 3, 300, 200

    #split dataset into train and test sets
    train, test = train_test_split(df, test_size=0.2)
    #reset the indexes of train and test sets
    train = train.reset_index()
    test = test.reset_index()


    padding = vectors["the"]
    padding = np.zeros(d)

    #Put questions from train set into format to calculate z vectors
    questions1 = train['question1']
    questions2 = train['question2']
    questionPairs = np.array([questions1, questions2])
    questions = flattenData(questionPairs)
    tokenized_Questions = tokenizeSentences(questions)
    #get the questions into their respective vectors with necessary padding
    vecInputs1, vecInputs2 = getEmbeddings(questions1, questions2, vectors, padding, k)
    #Get training set labels and Z's
    trainLabels = torch.from_numpy(train['is_duplicate'].values).float()
    trainZs = buildAllZs(vecInputs1, vecInputs2, k)

    #Put questions from test set into format to calculate z vectors
    questions1 = test['question1']
    questions2 = test['question2']
    questionPairs = np.array([questions1, questions2])
    questions = flattenData(questionPairs)
    tokenized_Questions = tokenizeSentences(questions)
    #get the questions into their respective vectors with necessary padding
    vecInputs1, vecInputs2 = getEmbeddings(questions1, questions2, vectors, padding, k)
    #Get test set labels and Z's
    testLabels = torch.from_numpy(test['is_duplicate'].values).float()
    testZs = buildAllZs(vecInputs1, vecInputs2, k)

    maxEpoch, threshold = 100, 0.005
    CNN = trainCNN(trainZs, trainLabels, testZs, testLabels, clu, k, d, maxEpoch, threshold)
    #testCNN()

    #outputs = buildInputsToCNN(vecInputs1, vecInputs2, labels, d, k, clu, len(questions1))
