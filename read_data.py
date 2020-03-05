import os
from nltk.tree import Tree


def split_file(filepath, write = False):
    """ Separate dataset in filepath in different files  """

    with open(filepath,"r") as f:
        lines = f.readlines()
    
    nbLines = len(lines)

    trainLines = lines[:int(0.8*nbLines)]
    testLines = lines[int(0.8*nbLines):int(0.9*nbLines)]
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
    """ Delete "-" from the input tree """

    if tree.label().find('-'):
        tree.set_label(tree.label().split('-')[0])
    for child in tree:
        if isinstance(child, Tree):
            clean_data(child)
    
# def print_data(tree):
#     print(tree.label())
#     for child in tree:
#         if isinstance(child, Tree):
#             print_data(child)
#         else :
#             print("Child", child)


def get_terminal(tree):
    """ Get terminal liaisons """
    leaves = []
    for child in tree :
        if isinstance(child,Tree):
            leaves.extend(get_terminal(child))
        else :
            leaves.append((child,tree.label()))

# def get_nonterminals(tree):
    # rules = []
    # subtrees = tree.subtrees()
    # leaf = False
    # for sub in subtrees :
    #     sonLabel = []
    #     for child in sub :
    #         if isinstance(child,Tree):
    #             sonLabel.append(child.label())
    #         else :
    #             leaf =True
    #             break
    #     if not leaf and len(sub.label())>0 :
    #         rules.append((sub.label(),sonLabel))
    # return rules

# def get_production(tree):
#     productions

def data_to_tree(filepath):
    listTrees = []
    with open(filepath,"r") as f:
        lines = f.readlines()

    for line in lines:
        tree = Tree.fromstring(line)
        clean_data(tree) # Delete '-' from labels
        tree.chomsky_normal_form()
        listTrees.append(tree)
    return listTrees