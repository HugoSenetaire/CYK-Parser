from read_data import *
from collections import defaultdict
from nltk import *
from nltk.tree import Tree
import numpy as np

class PCFG():
    def __init__(self):
        self.rules = {}
        self.rule_count = defaultdict(int)

        self.lexicon = {}
        self.lexicon_count = defaultdict(int)
    
    def learn(self,listTree):
        # print(listTree)
        for tree in listTree:
            for production in tree.productions():
                if production.is_lexical() :
                    if str(production.rhs()[0]) not in self.lexicon:
                        self.lexicon[str(production.rhs()[0])] = defaultdict(float) 
                    self.lexicon[str(production.rhs()[0])][(production.lhs(),)]+=1.
                    self.lexicon_count[str(production.rhs()[0])] +=1.
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
            

        # print(self.lexicon)
        
        # compteur = 0
        # for left in self.rules.keys():
        #     print("left",left)
        #     print("left", type(left))
        #     # print(str(left))
        #     # print(left.symbol())
        #     for right in self.rules[left].keys():
        #         print("right",right)
        #         print(type(right))
                
        #     # for right in self.rules[left]:
        #         # print(right[0])
        #     compteur+=1
        #     if compteur == 10 :
        #         break

    def CYK(self,sentence):
        words = sentence.split(" ")
        diagonal_table = []

        back = [[defaultdict() for i in range(len(words)-i)]for i in range(len(words))] 
        log_pr = [[{} for i in range(len(words)-i)]for i in range(len(words))] 
        # print(len(log_pr))
        diagonal_table.append([])
        for k in range(0,len(words)):
            diagonal_table[0].append([])
            for s in self.lexicon[words[k]]:
               
                diagonal_table[0][k]=[s]
                # print(diagonal_table)
                log_prob = np.log(self.lexicon[words[k]][s])
                log_pr[0][k][s] = (log_prob,0,k,words[k])
                back[0][k][s]= (0,k,words[k])
                ## PROBLEME AVEC LE BACK, ON POURRAIT EN AVOIR DEUX DANS LE MEME LIGNE SUR 2 COLONNES DIFFERENTES IDENTIQUES
                
                
        # print(back[0])
        for i in range(1,len(words)): # place in the pyramid i=1 : taille 2
            diagonal_table.append([])
            for init in range(len(words)-i): # Place in the row of the pyramid
                diagonal_table[i].append([])
                for length_init in range(1,i+1):
                    length_end = (i+1)-length_init
                    init_list = diagonal_table[length_init-1][init]
                    end_list = diagonal_table[length_end-1][init+length_init]
                    for element1 in init_list :
                        for element2 in end_list :
                            inputElement = element1+element2
                            if inputElement in self.rules :
                                for s in self.rules[inputElement]: # A priori proba non nul
                                    if self.rules[inputElement][s] == 0:
                                        currentProb =- float("inf")
                                    else :
                                        currentProb = log_pr[length_init-1][init][element1][0] \
                                            + log_pr[length_end-1][init+length_init][element2][0] \
                                            + np.log(self.rules[inputElement][s])
                                    if s not in back[i][init] :
                                        log_pr[i][init][s] = (currentProb, length_init-1, init, element1, length_end-1, init+length_init, element2)
                                        back[i][init][s] = (length_init-1, init, element1, length_end-1, init+length_init, element2)
                                        diagonal_table[i][init].append(s)
                                    elif currentProb>log_pr[i][init][s][0] :
                                        log_pr[i][init][s] = (currentProb, length_init-1, init, element1, length_end-1, init+length_init, element2)
                                        back[i][init][s] = (length_init-1, init, element1, length_end-1, init+length_init, element2)
                                    # Attention ca marche pas car proba table est complètement pourri 
                                    # On a plein de proba et pas juste celle du chemin
                                    # Dans log proba, si on peut avoir deux fois la même lettre, on choisit celle avec le plus de proba
                                    # La lettre doit apparaitre dans le tableau log proba
        
        if (grammar.Nonterminal("SENT"),) not in back[-1][0].keys():
            return False


        else :
            tree = self.build_tree(back, i, 0, (grammar.Nonterminal("SENT"),))
            return tree
            


    def build_tree(self, back, i, j, value):
        list_children = []
        list_back = back[i][j][value]
        if i > 0:
            for element in list_back[2]:
                list_children.append(self.build_tree(back,list_back[0], list_back[1], (element,))) #Problem si element est pas un tuple ? A modifier :
            for element in list_back[5]:
                list_children.append(self.build_tree(back,list_back[3], list_back[4], (element,)))
        else :
            list_children = [back[i][j][value][2]]
        tree = Tree(value[0],list_children)
        return tree
        

                        
                

            




 



