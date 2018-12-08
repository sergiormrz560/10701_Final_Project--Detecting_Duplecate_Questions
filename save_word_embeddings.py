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
    #df = df.head(50)

    #Put dataset into format to calc word embeddings
    questions1 = df['question1']
    questions2 = df['question2']
    questionPairs = np.array([questions1, questions2])
    questions = flattenData(questionPairs)
    tokenized_Questions = tokenizeSentences(questions)
    

    vectors = modelWord2Vec(tokenized_Questions, d)
    vectors.save("word2vec.model")
    
    #pcaVisulaization(model)

 