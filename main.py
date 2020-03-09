
import argparse
from read_data import *
from PCFG import *
from OOV import *
from evaluation import *
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
    listTrees_val = data_to_tree("Data/val")
    listTrees_test = data_to_tree("Data/test")
    pcfg = PCFG()
    pcfg.learn(listTrees)

##Embeddings :
    words, embeddings = pickle.load(open('Data/polyglot-fr.pkl', 'rb'), encoding='latin1')
    # Map words to indices and vice versa
    word_id = {w:i for (i, w) in enumerate(words)}
    id_word = dict(enumerate(words))
    # Put embeddings in PCFG
    pcfg.add_embedding(embeddings, word_id, id_word)
    

##Simple Test
    # print(pcfg.CYK("Je apprend des expositions ."))
    # print(pcfg.CYK("une pomme mange des pommes ."))
    # print(pcfg.CYK("des pomme mange des pommes ."))
    
    # print(pcfg.CYK("Le réseau C repose sur Edmond Kwan et Alfred Sirven , dirigeant d' Elf .",verbose = True))
    # print(listTrees_test[0].pretty_print())
    # print(pcfg.CYK(list(listTrees_test[0].flatten()),verbose = True))
    # print(listTrees_val[0].pretty_print())
    # print(listTrees_val[0])
    # print(pcfg.CYK(list(listTrees_val[0].flatten()),verbose = True))

    # aux = "- Gérard Longuet"
    # aux = 'Affaire politico- financière'
    # aux = 'Le procès en première instance'
    # aux = "- Juin 2005 : l', ex- conseiller général RPR des Hauts-de-Seine Didier Schuller et le député-maire -LRB- UMP -RRB- de Levallois-Perret Patrick Balkany comparaissent devant le tribunal correctionnel de Créteil dans l' affaire des HLM des Hauts-de-Seine ."
    # aux = "- Juin 2005 : l', ex- conseiller général RPR des Hauts-de-Seine Didier Schuller comparaissent devant le tribunal correctionnel de Créteil dans l' affaire des HLM des Hauts-de-Seine ."
    # aux = "La droite crie au scandale et affirme que c' est Chirac que l'_on cherche à atteindre au travers de son ministre de l' Intérieur de l' époque ."
    # aux = "Bibliographie"
    # aux = "Le exposition apprend"
    # aux = "Les ascenseurs des HLM de Paris"
    # aux = "Les marchés truqués"
    # aux = "Ils se sont assurés que l' Irak ne dispose d' aucune fusée à longue portée pouvant menacer ses voisins ."
    # print(pcfg.CYK(aux, verbose =True))
## Global evaluation :
    # evaluate_parser(pcfg, listTrees_test, filepath="parser_output.txt", write = False)
    evaluate_parser(pcfg, listTrees_val, filepath="parser_output.txt", write = True)
