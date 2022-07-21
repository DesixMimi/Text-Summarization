import numpy as np
import math
from collections import Counter
import sacrebleu                    # import sacrebleu in order compute the BLEU score.
import matplotlib.pyplot as plt

import wordcloud
from nltk.util import ngrams
import nltk
from nltk.translate.bleu_score import sentence_bleu
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('omw-1.4')

import warnings
warnings.filterwarnings("ignore")
from nlp_utils import *
import datasets
import pandas as pd
import numpy as np
import keras
import re
import datasets
import contractions
import tensorflow as tf
import tensorflow_datasets as tfds

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer 
# from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords   
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K


import warnings

