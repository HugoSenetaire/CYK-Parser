import numpy as np

def damereau_levenshtein_distance(sinput, soutput):
    m = np.zeros(len(sinput),len(soutput))
    for i in range(len(sinput)):
        m[i,0] = 0
    for j in range(len(soutput)):
        m[0,j] = 0
    
    for i in range(1, len(sinput)):
        for j in range(1,len(soutput)):
            elif sinput[i]==soutput[j]:
                m[i,j] = min(m[i-1,j]+1,m[i,j-1]+1, m[i-1,j-1])
            else :
                m[i,j] = min(m[i-1,j]+1,m[i,j-1]+1, m[i-1,j-1]+1)
            if i>1 and j>1 :
                m[i,j] = min(m[i,j], m[i-2,j-2]+1)
    return m[-1,-1]