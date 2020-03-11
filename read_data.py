import os
from nltk.tree import Tree
import numpy as np



def split_file(filepath, write = False):
    """ Separate dataset in filepath in different files  """

    with open(filepath,"r") as f:
        lines = f.readlines()
    
    nbLines = len(lines)
    trainLines = lines[:int(0.9*nbLines)]
    testLines = lines[int(0.8*nbLines):int(0.9*nbLines)]
    # aux = lines[:int(0.9*nbLines)]
    # np.random.shuffle(aux)
    # trainLines = aux[:int(0.8*nbLines)]
    # testLines = aux[int(0.8*nbLines):int(0.9*nbLines)]
    valLines = lines[int(0.9*nbLines):]

    if write :
        with open("Data/train", "w") as f :
            f.writelines(trainLines)
        with open("Data/test", "w") as f :
            f.writelines(testLines)
        with open("Data/val", "w") as f :
            f.writelines(valLines)
    
    return trainLines, testLines, valLines


def clean_data(tree):
    """ Delete "-" from the input tree (functionnal)
    :param tree:
    
    """

    if tree.label().find('-'):
        tree.set_label(tree.label().split('-')[0])
    for child in tree:
        if isinstance(child, Tree):
            clean_data(child)
    

def get_terminal(tree):
    """ Get terminal symbol for the tree """
    leaves = []
    for child in tree :
        if isinstance(child,Tree):
            leaves.extend(get_terminal(child))
        else :
            leaves.append((child,tree.label()))


def clean_leaves(tree, symbol = "&"):
    """ clean leaves from the collapsing symbol 
    example : A->B->"word" => unary => A&B->"word" => clean_leaves => A->"word"
    :param tree: input tree
    :param symbol: symbol used for collapse (default is "&")

    """
    change = False
    for child in tree:
        if isinstance(child, Tree):
            clean_leaves(child)
        else:
            change = True
    if change :
        newLabel = tree.label().split(symbol)[0]
        tree.set_label(newLabel)    


def data_to_tree(filepath):
    """ Read the data and turn it into a list of normalized Chomsky tree
    :param filepath: input txt file
    :return listTrees: list of output trees"""
    listTrees = []
    with open(filepath,"r", encoding = 'utf-8') as f:
        lines = f.readlines()

    for line in lines:
        tree = Tree.fromstring(line)
        clean_data(tree[0]) # Delete '-' from labels
        tree.collapse_unary(collapsePOS = True,joinChar = "&")
        tree.chomsky_normal_form(horzMarkov=2)
        listTrees.append(tree)
    return listTrees