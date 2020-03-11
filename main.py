
import argparse
from read_data import *
from PCFG import *
from OOV import *
from evaluation import *
import numpy as np
import pickle


if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--sentence','-s', type=str, default = None, help='sentence to parse')
    parser.add_argument('--mode','-m',type=str, default = 'eval')
    parser.add_argument('--embedding', '-e', type=str, default = 'Data/polyglot-fr.pkl')
    parser.add_argument('--dataset_val', '-d', type = str, default = 'Data/val')
    parser.add_argument('--write', '-w',  default = False)
    args = parser.parse_args()

## Read Files and create pcfg :
    split_file("Data/sequoia-corpus+fct.mrg_strict", write = True)
    listTrees = data_to_tree("Data/train")
    listTrees_val = data_to_tree(args.dataset_val)
    listTrees_test = data_to_tree("Data/test")
    pcfg = PCFG()
    pcfg.learn(listTrees)

##Embeddings :
    words, embeddings = pickle.load(open(args.embedding, 'rb'), encoding='latin1')
    # Map words to indices and vice versa
    word_id = {w:i for (i, w) in enumerate(words)}
    id_word = dict(enumerate(words))
    # Put embeddings in PCFG
    pcfg.add_embedding(embeddings, word_id, id_word)
    
##Simple Test
    if args.mode == 'eval':
        if args.sentence is None :
            print("You must give a sentence to parse as option (use -s) ")
        else :
            for sentence in args.sentence.split("\n"):
                tree = pcfg.CYK(sentence, verbose = True)
                if not tree :
                    print("The CYK was not able to parse the sentence '{}' ".format(sentence))
                else :
                    print("The tree for the sentence '{}'".format(sentence))
                    tree.un_chomsky_normal_form(unaryChar="&")
                    print(tree)
   
## Global evaluation :
    if args.mode == "dataset":
        evaluate_parser_multiprocess(pcfg, listTrees_val, filepath=args.dataset_val.split(".")[0] + "output.txt", write = args.write)
        