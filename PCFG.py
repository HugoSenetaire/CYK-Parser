from read_data import *
from collections import defaultdict
from nltk import *
from nltk.tree import Tree
import numpy as np
from OOV import *
from nltk.tokenize import word_tokenize


def check_Sent(liste_output):
    """ Check the list on the higher row of the pyramid to get the start symbol of the sentence parsing
    :param list_output: list of words on the higher row of pyramid """
    elementTrue = None
    proba = - float("inf")
    result = False
    for element in liste_output.keys():
        if str(element[0]).startswith("SENT"):
            result=True 
            if liste_output[element][0]>proba:
                proba = liste_output[element][0]
                elementTrue = element

    return result, elementTrue

class PCFG():
    def __init__(self):
        self.rules = {}
        self.rule_count = defaultdict(int)

        self.lexicon = {}
        self.lexicon_count = defaultdict(int)




        self.embeddings = None
        self.word_id = {}
        self.id_word = {}
    
    def learn(self,listTree):
        """
        Training the PCFG to get the different rules of the CFG
        :param listTree: Tree to train the PCFG from
        """
        for tree in listTree:
            for production in tree.productions():
                if production.is_lexical() :
                    if str(production.rhs()[0]) not in self.lexicon:
                        self.lexicon[str(production.rhs()[0])] = defaultdict(float)


                    if "&" not in str(production.lhs()) or len(str(production.lhs()))<2 :
                        self.lexicon[str(production.rhs()[0])][(production.lhs(),)]+=10.
                        self.lexicon_count[str(production.rhs()[0])]+=10
                    else :
                        B = str(production.lhs()).split("&")[-1]
                        # A = str(production.lhs()).strip("&"+B)
                        # A = (grammar.Nonterminal(A),)
                        B = (grammar.Nonterminal(B),)

                        # if A not in self.lexicon :
                            # self.lexicon[A] = defaultdict(float)
                        if B not in self.lexicon :
                            self.lexicon[B] = defaultdict(float)

                        self.lexicon[str(production.rhs()[0])][(production.lhs(),)]+=8.
                        # self.lexicon[str(production.rhs()[0])][A]+=1.
                        self.lexicon[str(production.rhs()[0])][B]+=2.
                        self.lexicon_count[str(production.rhs()[0])]+=10
                    
                else :
                    if len(str(production.lhs()))>0:
                        if production.rhs() not in self.rules :
                            self.rules[production.rhs()] = defaultdict(float) 
                        self.rules[production.rhs()][(production.lhs(),)]+=1.
                        self.rule_count[production.rhs()] +=1.
        
        for left in self.rules.keys():
            for right in self.rules[left].keys():
                self.rules[left][right] /= float(self.rule_count[left])

        for left in self.lexicon.keys():
            for right in self.lexicon[left].keys():
                self.lexicon[left][right] /= float(self.lexicon_count[left])
            
        nbRule = 0
        for key in self.rules :
            nbRule+=len(self.rules[key].keys())
        print("We have {}  rules".format(nbRule))

    def add_embedding(self,embeddings, word_id, id_word):
        """
        Add embeddings caracteristics to PCFG class
        :param embeddings:
        :param word_id: dict word->id
        :param id_word: dict id -> word
        """
        self.embeddings = embeddings
        self.word_id = word_id
        self.id_word = id_word



    def CYK(self,sentence,verbose = False):
        """
        Parse a sentence or a table of words to get the more likely graph
        :param sentence: list of words
        :param verbose: bool print data
        """

        if isinstance(sentence, str):
            words = sentence.split(" ")
        else :
            words = sentence


        if verbose :
            print("Original sentence :")
            print(sentence)
        
        self.dic_replacement = {}
        new_sentence = ""



        back = [[{} for i in range(len(words)-i)] for i in range(len(words))] 
        log_pr = [[{} for i in range(len(words)-i)] for i in range(len(words))] 



        for k in range(0,len(words)):
            # Treating OOV :
            if words[k] not in self.lexicon :
                liste = closest(words[k], self.lexicon, self.embeddings,  self.word_id, self.id_word)
                if verbose :
                    print(f"{words[k]} is replaced by {liste}")
            else :
                liste = [(words[k],1.0)]
                if words[k][0].isupper() and words[k].lower() in self.lexicon :
                      liste.append((words[k].lower(),0.5)) # Try the lowercase with lower probability
            

            # Create the auxiliary sentence that will be treated
            for i,element in enumerate(liste) :
                word = element[0]
                if i == 0:
                    new_sentence+=word
                else :
                    new_sentence+="/" + word 


            # Create the first line with self.lexicon
            for element in liste :
                proba = element[1]
                word = element[0]

                for s in self.lexicon[word]:
                    currentProb = np.log(self.lexicon[word][s]) + np.log(proba)
                    self.dic_replacement[k,word] = words[k]
                    if s not in back[0][k] :
                        log_pr[0][k][s] = (currentProb,0,k,word)
                        back[0][k][s] = (0,k,word)

                    elif currentProb>log_pr[0][k][s][0] :
                        log_pr[0][k][s] = (currentProb,0,k,word)
                        back[0][k][s] = (0,k,word)
            
            new_sentence+=" "    


        if verbose :
            print("Auxiliary sentence :")
            print(new_sentence)
        

        # If only one words, we reconstruct the tree :
        if len(words)==1:
            result, element = check_Sent(log_pr[0][0])
            if result :
                origin,pos = str(element[0]).split("&",1)
                tree = Tree(origin,[Tree(pos,[back[0][0][element][2]])])
                return tree
            else :
                elementTrue = None
                proba = -float("inf")
                for element in back[0][0].keys():
                    if back[0][0][element][0]>proba:
                        proba = back[0][0][element][0]
                        elementTrue = element
                tree = Tree("SENT", [Tree(str(elementTrue[0]),[back[0][0][element][2]])])
                return tree

        # Check the upper cells of the pyramid
        for i in range(1,len(words)): # Place in the row of the pyramid (lenght is i+1)
            for init in range(len(words)-i): # Place in the column of the pyramid 
                for length_init in range(1,i+1): # Where to split in the sentence
                    length_end = (i+1)-length_init
                    init_list = back[length_init-1][init].keys()
                    end_list = back[length_end-1][init+length_init].keys()
                    for element1 in init_list :
                        for element2 in end_list :
                            inputElement = element1+element2
                            if inputElement in self.rules :
                                for s in self.rules[inputElement]:
                                    if self.rules[inputElement][s] == 0: # There should be no element here
                                        currentProb =- float("inf")
                                    else :
                                        currentProb = log_pr[length_init-1][init][element1][0] \
                                            + log_pr[length_end-1][init+length_init][element2][0] \
                                            + np.log(self.rules[inputElement][s])
                                    if s not in back[i][init] :
                                        log_pr[i][init][s] = (currentProb, length_init-1, init, element1, length_end-1, init+length_init, element2)
                                        back[i][init][s] = (length_init-1, init, element1, length_end-1, init+length_init, element2)
                                    elif currentProb>log_pr[i][init][s][0] :
                                        log_pr[i][init][s] = (currentProb, length_init-1, init, element1, length_end-1, init+length_init, element2)
                                        back[i][init][s] = (length_init-1, init, element1, length_end-1, init+length_init, element2)
                                


        resultParsable, element = check_Sent(log_pr[-1][0])
        if not resultParsable:
            return False
        else :
            tree = self.build_tree(back, i, 0, element)
            tree.set_label("SENT")
            return tree
            


    def build_tree(self, back, i, j, value):
        """
        build_tree from the pyramid table backs
        :param back: dictionnary for the pyramid
        :param i: row in the pyramid
        :param j: column in the pyramid
        :param value: symbol we are looking for
        :return tree: tree of the pyramid
        """
        list_children = []
        list_back = back[i][j][value]
        if i > 0:
            for element in list_back[2]: # First symbol
                list_children.append(self.build_tree(back,list_back[0], list_back[1], (element,)))
            for element in list_back[5]: # Second symbol
                list_children.append(self.build_tree(back,list_back[3], list_back[4], (element,)))
        else :
            list_children = [self.dic_replacement[j,back[i][j][value][2]]]
        tree = Tree(str(value[0]),list_children)
        return tree
        
