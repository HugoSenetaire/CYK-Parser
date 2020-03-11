
from operator import itemgetter
import re
import numpy as np

def damereau_levenshtein_distance(sinput, soutput):
    """ Get the edit distance between sinput and soutput
    :param sinput: string input
    :param soutput: string output to compare
    :return distance: 

    """
    m = np.zeros((len(sinput),len(soutput)))
    for i in range(len(sinput)):
        m[i,0] = i
    for j in range(len(soutput)):
        m[0,j] = j
    
    for i in range(1, len(sinput)):
        for j in range(1,len(soutput)):
            if sinput[i]==soutput[j]:
                m[i,j] = min(m[i-1,j]+1,m[i,j-1]+1, m[i-1,j-1])
            else :
                m[i,j] = min(m[i-1,j]+1,m[i,j-1]+1, m[i-1,j-1]+1)
            if i>1 and j>1 and sinput[i-1]==soutput[j] and sinput[i]==soutput[j-1] :
                m[i,j] = min(m[i,j], m[i-2,j-2]+1)
    return m[-1,-1]


def case_normalizer(word, dictionary):
  """ In case the word is not available in the vocabulary,
     we can try multiple case normalizing procedure.
     We consider the best substitute to be the one with the lowest index,
     which is equivalent to the most frequent alternative."""
  w = word
  lower = (dictionary.get(w.lower(), 1e12), w.lower())
  upper = (dictionary.get(w.upper(), 1e12), w.upper())
  title = (dictionary.get(w.title(), 1e12), w.title())
  results = [lower, upper, title]
  results.sort()
  index, w = results[0]
  if index != 1e12:
    return w
  return word


def normalize(word, word_id):
    """ Find the closest alternative in case the word is OOV."""
    DIGITS = re.compile("[0-9]", re.UNICODE)
    if not word in word_id:
        word = DIGITS.sub("#", word)
    if not word in word_id:
        word = case_normalizer(word, word_id)

    if not word in word_id:
        return None


    return word


def closest(word, lexicon, embeddings = None, word_id = {}, id_word = {}, possibleNPP = True, maxembedd = 5):
    """ Get maxembedd closest elements from embeddings if word is in embedding. Otherwise give the word with the smallest edit distance from lexicon
        Return a list of tuple (word, coefficient)
    """
    candidates = []
    if word[0].isupper() and possibleNPP : # Check for NPP
        candidates.append(('Cora',0.01))
        
    wordNew = normalize(word, word_id)
    if (not wordNew) or (embeddings is None):
        best_dist = float("inf")
        best_word = None
        for word_test in lexicon :
            dist = damereau_levenshtein_distance(word,word_test)
            if dist < best_dist :
                best_dist = dist 
                best_word = word_test
        candidates.append((best_word,1.0))
        return candidates

    else :
        word = wordNew
        word_index = word_id[word]
        e = embeddings[word_index]
        distance = [] 
        for word_test in lexicon :
            if word_test in word_id :
                word_test_index = word_id[word_test]
                distance.append((word_test,-embeddings[word_test_index].dot(e)/np.linalg.norm(embeddings[word_test_index])/np.linalg.norm(e)))
        distance = sorted(distance,key=itemgetter(1))
        for k in range(maxembedd):
            candidates.append((distance[k][0],1.0))
        return candidates



        

    
