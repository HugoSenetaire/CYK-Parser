
import argparse
from read_data import *
from PCFG import *
from OOV import *
import numpy as np
import pickle


if __name__ == "__main__" :

    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('sentence', type=str,
    #                     help='sentence to parse')
    # parser.add_argument('train', type=str, default = 'Data/train'
    #                     help='file to train')

    # args = parser.parse_args()

## Read Files and create pcfg :
    split_file("Data/sequoia-corpus+fct.mrg_strict", write = True)
    listTrees = data_to_tree("Data/train")
    pcfg = PCFG()
    pcfg.learn(listTrees)

##Embeddings :
    words, embeddings = pickle.load(open('Data/polyglot-fr.pkl', 'rb'), encoding='latin1')
    print(np.shape(embeddings))
    # Map words to indices and vice versa
    word_id = {w:i for (i, w) in enumerate(words)}
    id_word = dict(enumerate(words))
    
    


    print(pcfg.CYK("Je apprend des expositions ."))

