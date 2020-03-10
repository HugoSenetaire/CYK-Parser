from read_data import *
import numpy as np
from nltk.grammar import Nonterminal
from multiprocessing import *
import multiprocessing
from PYEVALB import scorer
from PYEVALB import parser
import time

# target = "( (SENT (PP (P Au_cours_de) (NP (DET la) (NP|<NC-PP> (NC cérémonie) (PP (P d') (NP+NC inauguration))))) (PONCT .)))"
# predicted = "( (SENT+PP (P Au_cours_de) (NP (DET la) (NP|<NC-PP> (NC cérémonie) (NP|<PP-PONCT> (PP (P d') (NP+NC inauguration)) (PONCT .))))))"



def accuracy(list_pos_predicted, list_pos_true):
    """
    :param list_pos_predicted:
    :param list_pos_true:
    :return: Accuracyc
    """
    return (np.array(list_pos_predicted)==np.array(list_pos_true)).sum()/len(list_pos_predicted)


def get_leaves(tree, label = None):
    """
    Return list with leaves token and label(POS)
    :param tree
    :param label:
    :return list of label leaves:
    """
    leaves = []
    for child in tree:
        if isinstance(child, Tree):
            leaves.extend(get_leaves(child, child.label()))
        else:
            if isinstance(label,str):
                leaves.append(Nonterminal(label))
            else :
                leaves.append(label)

    return leaves





def evaluate_parser(pcfg, test_trees, filepath="parser_output.txt", write = True):
    """
    Method to evaluate the parser
    :param pcfg: parser pcfg to evaluate
    """
   
    y_true = []
    y_pred = []

    y_true_non_chomsky = []
    y_pred_non_chomsky = []

    y_true_parsable = []
    y_pred_parsable = []

    y_true_parsable_non_chomsky = []
    y_pred_parsable_non_chomsky = []


    recall_list = []
    precision_list = []
    lines = []


    with open(filepath, 'w') as file:
        file.write("" )
    with open("non-parsable", 'w') as file:
        file.write("")

    nb_non_parsable = 0
    list_non_parsable = []
    for (c, tree) in enumerate(test_trees):
        print('{0:0.2f}% done'.format(100*c/len(test_trees)))
        test_sentence = list(tree.flatten())
        parsed_tree = pcfg.CYK(test_sentence,verbose = False)
        test_sentence_str = ' '.join(str(parsed_tree).split())
        print("Tested sentence{}".format(test_sentence))

        if parsed_tree:
            
            y_true.extend(get_leaves(tree))
            y_pred.extend(get_leaves(parsed_tree))
            y_true_parsable.extend(get_leaves(tree))
            y_pred_parsable.extend(get_leaves(parsed_tree))


            tree.un_chomsky_normal_form(unaryChar="&")
            parsed_tree.un_chomsky_normal_form(unaryChar="&")
            y_true_non_chomsky.extend(get_leaves(tree))
            y_pred_non_chomsky.extend(get_leaves(parsed_tree))
            y_true_parsable_non_chomsky.extend(get_leaves(tree))
            y_pred_parsable_non_chomsky.extend(get_leaves(parsed_tree))
            parsed_tree_str = ' '.join(str(parsed_tree).split())
            lines.append('( '+' '.join(str(parsed_tree).split()) + ')')

            print('Nb Non parsable {}'.format(nb_non_parsable))
            print('Accuracy total chomsky on dev set {}:'.format(accuracy(y_pred, y_true)))
            print("Accuracy total non chomsky on dev set {}:".format(accuracy(y_true_non_chomsky,y_pred_non_chomsky)))
            print('Accuracy parsable chomsky on dev set {}:'.format(accuracy(y_pred_parsable, y_true_parsable)))
            print("Accuracy parsable non chomsky on dev set {}:".format(accuracy(y_true_parsable_non_chomsky,y_pred_parsable_non_chomsky)))

            target_tree = parser.create_from_bracket_string(test_sentence_str) 
            predicted_tree = parser.create_from_bracket_string(parsed_tree_str) 

            s = scorer.Scorer() ## old
            try :
                result = s.score_trees(target_tree, predicted_tree)
                print('Precision =' + str(result.prec))
                print('Recall =' + str(result.recall))
            except :
                print("No Recall or precision")
            
            recall_list.append(result.recall)
            precision_list.append(result.prec)

            if write :
                with open(filepath, 'a') as file:
                    file.write(lines[-1]+"\n" )
        
        else :
            aux = get_leaves(tree)
            print("Sentence not parsable")
            y_true.extend(aux)
            y_pred.extend(["None"for k in range(len(aux))])

            tree.un_chomsky_normal_form(unaryChar="&")
            y_true_non_chomsky.extend(get_leaves(tree))
            y_pred_non_chomsky.extend(["None"for k in range(len(get_leaves(tree)))])

            nb_non_parsable+=1
            list_non_parsable.append(test_sentence)

            print('Accuracy total chomsky on dev set {}:'.format(accuracy(y_pred, y_true)))
            print('Accuracy total non chomsky on dev set {}:'.format(accuracy(y_pred_non_chomsky, y_true_non_chomsky)))
            print('Nb Non parsable {}'.format(nb_non_parsable))
            if write :
                with open(filepath, 'a') as file:
                    file.write("\n" )
                with open("non-parsable", 'a') as file:
                    file.write('( '+' '.join(str(tree).split()) + ')' + "\n")
            

    print('Nb Non parsable {}'.format(nb_non_parsable))
    print('Accuracy total chomsky on dev set {}:'.format(accuracy(y_pred, y_true)))
    print("Accuracy total non chomsky on dev set {}:".format(accuracy(y_true_non_chomsky,y_pred_non_chomsky)))
    print('Accuracy parsable chomsky on dev set {}:'.format(accuracy(y_pred_parsable, y_true_parsable)))
    print("Accuracy parsable non chomsky on dev set {}:".format(accuracy(y_true_parsable_non_chomsky,y_pred_parsable_non_chomsky)))
    print("Recall moyen {} et précision moyenne {}".format(np.mean(recall_list),np.mean(precision_list)))
    



def evaluate_parser_multiprocess(pcfg, test_trees, filepath="parser_output.txt", write = True):
    """
    Method to evaluate the parser
    :param pcfg: parser pcfg to evaluate
    """
   
    y_true = []
    y_pred = []

    y_true_non_chomsky = []
    y_pred_non_chomsky = []

    y_true_parsable = []
    y_pred_parsable = []

    y_true_parsable_non_chomsky = []
    y_pred_parsable_non_chomsky = []

    recall_list = []
    precision_list = []
    lines = []


    with open(filepath, 'w') as file:
        file.write("" )
    with open("non-parsable", 'w') as file:
        file.write("")

    list_sentence = []
    for c,tree in enumerate(test_trees):
        list_sentence.append(list(tree.flatten()))

    # Parsing multi_process :
    n_job = multiprocessing.cpu_count()
    start = time.time()
    with Pool(n_job) as p :
        result_trees = p.map(pcfg.CYK,list_sentence)
    print(f"Parsing time is {time.time()-start}")
  
    nb_non_parsable = 0
    list_non_parsable = []
    for (c, tree) in enumerate(test_trees):
        print('{0:0.2f}% done'.format(100*c/len(test_trees)))
        test_sentence = list(tree.flatten())
        parsed_tree = result_trees[c]
        test_sentence_str = ' '.join(str(parsed_tree).split())
        print("Tested sentence{}".format(test_sentence))

        if parsed_tree:
            
            y_true.extend(get_leaves(tree))
            y_pred.extend(get_leaves(parsed_tree))
            y_true_parsable.extend(get_leaves(tree))
            y_pred_parsable.extend(get_leaves(parsed_tree))

            parsed_tree_str = ' '.join(str(parsed_tree).split())
            

            tree.un_chomsky_normal_form(unaryChar="&")
            parsed_tree.un_chomsky_normal_form(unaryChar="&")
            y_true_non_chomsky.extend(get_leaves(tree))
            y_pred_non_chomsky.extend(get_leaves(parsed_tree))
            y_true_parsable_non_chomsky.extend(get_leaves(tree))
            y_pred_parsable_non_chomsky.extend(get_leaves(parsed_tree))
            lines.append('( '+' '.join(str(parsed_tree).split()) + ')')


          
            target_tree = parser.create_from_bracket_string(test_sentence_str) 
            predicted_tree = parser.create_from_bracket_string(parsed_tree_str) 

            s = scorer.Scorer() ## old
            try :
                result = s.score_trees(target_tree, predicted_tree)
                # print('Precision =' + str(result.prec))
                # print('Recall =' + str(result.recall))
            except :
                print("No Recall or precision")
            
            recall_list.append(result.recall)
            precision_list.append(result.prec)

            if write :
                with open(filepath, 'a') as file:
                    file.write(lines[-1]+"\n" )
        
        else :
            aux = get_leaves(tree)
            # print("Sentence not parsable")
            y_true.extend(aux)
            y_pred.extend(["None"for k in range(len(aux))])

            tree.un_chomsky_normal_form(unaryChar="&")
            y_true_non_chomsky.extend(get_leaves(tree))
            y_pred_non_chomsky.extend(["None"for k in range(len(get_leaves(tree)))])

            nb_non_parsable+=1
            list_non_parsable.append(test_sentence)

            # print('Accuracy total chomsky on dev set {}:'.format(accuracy(y_pred, y_true)))
            # print('Accuracy total non chomsky on dev set {}:'.format(accuracy(y_pred_non_chomsky, y_true_non_chomsky)))
            # print('Nb Non parsable {}'.format(nb_non_parsable))
            if write :
                with open(filepath, 'a') as file:
                    file.write("\n" )
                with open("non-parsable", 'a') as file:
                    file.write('( '+' '.join(str(tree).split()) + ')' + "\n")
            

    print('Nb Non parsable {}'.format(nb_non_parsable))
    print('Accuracy total chomsky on dev set {}:'.format(accuracy(y_pred, y_true)))
    print("Accuracy total non chomsky on dev set {}:".format(accuracy(y_true_non_chomsky,y_pred_non_chomsky)))
    print('Accuracy parsable chomsky on dev set {}:'.format(accuracy(y_pred_parsable, y_true_parsable)))
    print("Accuracy parsable non chomsky on dev set {}:".format(accuracy(y_true_parsable_non_chomsky,y_pred_parsable_non_chomsky)))
    print("Recall moyen {} et précision moyenne {}".format(np.mean(recall_list),np.mean(precision_list)))
    



def evaluate_lexicon(PCFGParser, embeddings, test_trees):
    """
    Method to evaluate the probabilized lexicon only.(Greedy search)
    :return:
    """

    print('Nb rules: {}'.format(len(par.non_terminal_rules_counts.keys())))
    print('Nb terminal rules: {}'.format(len(par.lexicon_rules_counts.keys())))
    y_true = []
    y_pred = []
    test_trees = ListTrees('data/test.txt')

    for (c, tree) in enumerate(test_trees):
        print('{0:0.2f}% done'.format(100*c/len(test_trees)))
        tree.un_chomsky_normal_form(unaryChar='&')
        test_sentence = ovo.modify_tokens(list(tree.flatten()), False)
        parsed_labels = lex.predict_sentence(test_sentence)
        print(parsed_labels)
        print(leaves_labels(tree))
        y_true.extend(leaves_labels(tree))
        y_pred.extend(parsed_labels)

        print('Accuracy on dev set {}:'.format(accuracy(y_pred, y_true)))

    print('Accuracy on dev set {}:'.format(accuracy(y_pred, y_true)))
    