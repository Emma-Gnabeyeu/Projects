import itertools
import numpy as np
import math as m
import time
import random

test_points=[(5,0,1), (-1,-3,4), (-1,-4,-3), (-1,4,-3), (4,-3,0), (0,5,1),(7,3,-1), (2,0,-2)]

def dist(x,y):
    return m.sqrt(sum([(x[i]-y[i])**2 for i in range(len(x))]))

def circum(p):
    d=len(p[0])
    k=len(p)
    a=np.array([[p[i][j]-p[0][j]  for j in range(d)]+[0 for l in range(k-1)] for i in range(1,k)]+[[0 for i in range(j)]+[1]+[0 for i in range(d-j-1)]+[-p[i][j]+p[k-1][j] for i in range(k-1)] for j in range(d)])
    b=np.array([sum([(p[i][j]*p[i][j]-p[0][j]*p[0][j])/2 for j in range(d)]) for i in range(1,k)]+[p[k-1][i] for i in range(d)])
    c=()
    try:
        x = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
         return c,-1
    for i in range(d):
        c+=(sum([x[d+j]*(p[j][i]-p[k-1][i]) for j in range(k-1)])+p[k-1][i],)
    return c,dist(c,p[0])

def MEB_aux(points, base,d):
    if len(points)==0 or len(base)==d+1:
        return base
    x=random.choice(points)
    while x in base:
        points.remove(x)
        x=random.choice(points)
    points.remove(x)
    D=MEB_aux(points, base,d)
    if D!=[] and dist(x,circum(D)[0])<=circum(D)[1]:
        return D
    base.append(x)
    return MEB_aux(points, base,d)
        

def MEB(points):
    d=len(points[0])
    return circum(MEB_aux(points, [], d))

def max_cech(points,k):
    fichier = open("coord.txt", "w")
    for i in points:
        z=list(i)
        for r in range(len(z)):
            z[r]=str(z[r])
        fichier.write(' '.join(z)+'\n')
    fichier.close()
    x=[i for i in range(len(points))]
    simplexes=[{} for i in range(k+1)]
    fichier = open("cplx.txt", "w")
    for i in range(k+1):
        for j in itertools.combinations(x,min(i+1,len(x))):
            d=(MEB([points[k] for k in j]))[1]
            if d>=0:
                simplexes[i][j]=d
                z=list(j)
                for r in range(len(z)):
                    z[r]=str(z[r])
                fichier.write(' '.join(z)+'\n')
    fichier.close()
    return simplexes

def naive_cech(points,k, l):
    fichier = open("coord.txt", "w")
    for i in points:
        z=list(i)
        for r in range(len(z)):
            z[r]=str(z[r])
        fichier.write(' '.join(z)+'\n')
    fichier.close()
    x=[i for i in range(len(points))]
    simplexes=[{} for i in range(k+1)]
    d=0
    fichier = open("cplx.txt", "w")
    for i in range(k+1):
        for j in itertools.combinations(x,min(i+1,len(x))):
            d=(MEB([points[k] for k in j]))[1]
            if d<=l and d>=0:
                simplexes[i][j]=d
                z=list(j)
                for r in range(len(z)):
                    z[r]=str(z[r])
                fichier.write(' '.join(z)+'\n')
    fichier.close()
    return simplexes

def print_naive_cech(points,k,l):
    simplexes=naive_cech(points,k,l)
    for i in range(k+1):
        for k, v in sorted(simplexes[i].items(), key=lambda x: x[1]):
            print("%s -> [%s]" % (k, v))
        
def cech(points, k, l):
    fichier = open("coord.txt", "w")
    for i in points:
        z=list(i)
        for r in range(len(z)):
            z[r]=str(z[r])
        fichier.write(' '.join(z)+'\n')
    fichier.close()
    fichier = open("cplx.txt", "w")
    x=[i for i in range(len(points))]
    d=0
    doublons=[]
    simplexes=[{} for i in range(k+1)]
    voisins=[{} for i in range(k+1)]
    for i in range(len(points)):
        simplexes[0][(i,)]=0
        voisins[0][(i,)]=set()
        fichier.write(str(i)+'\n')
    for i,j in itertools.combinations(x,2):
        d= (MEB([points[i],points[j]]))[1]
        if d<=l and d>=0 :
            simplexes[1][(i,j)]=d
            z=[i,j]
            for r in range(len(z)):
                z[r]=str(z[r])
            fichier.write(' '.join(z)+'\n')
            voisins[0][(i,)].add(j)
            voisins[0][(j,)].add(i)
    for i in simplexes[1].keys():
        voisins[1][i]=(voisins[0][(i)[0],]).intersection(voisins[0][(i)[1],])
    for i in range(2,k+1):
        doublons=[]
        for k,v in simplexes[i-1].items():
            for j in voisins[i-1][k]:
                d=MEB([points[m] for m in (k+(j,))])[1]
                if set(k+(j,)) not in doublons and d<=l and d>=0:
                    doublons+=[set(k+(j,))]
                    simplexes[i][k+(j,)]=MEB([points[m] for m in (k+(j,))])[1]
                    voisins[i][k+(j,)]=(voisins[i-1][k]).intersection(voisins[0][(j,)])
                    z=list(k+(j,))
                    for r in range(len(z)):
                        z[r]=str(z[r])
                    fichier.write(' '.join(z)+'\n')
    return simplexes
                    
def print_cech(points,k, l):
    simplexe = cech(points,k, l)
    for i in range(k+1):
        for k, v in sorted(simplexe[i].items(), key=lambda x: x[1]):
            print("%s -> [%s]" % (k, v))

"""
autre notion de voisin d'un k-simplexe possible: 
point qui forme un k-simplexe avec tous k-1 points du simplexe
avantage: plus restreignant
désavantage: plus difficile à mettre à jour
"""

def cech_comp_time(points, k, l):
    dt1=0
    dt2=0
    a=0
    for i in range(1000):
        t1=time.time()
        a=naive_cech(points, k, l)
        dt1+=time.time()-t1
        t2=time.time()
        a=cech(points, k, l)
        dt2+=time.time()-t2
    return dt1/1000,dt2/1000

def a_filt(V,S):
    o,r=circum(S)
    i=0
    while r!=-1 and i<len(V):
        if dist(V[i],o)<r:
            o,r=a_filt(V[:i],S+[V[i]])
        i+=1
    return o,r

def a_comp(points, k, l):
    fichier = open("coord.txt", "w")
    for i in points:
        z=list(i)
        for r in range(len(z)):
            z[r]=str(z[r])
        fichier.write(' '.join(z)+'\n')
    fichier.close()
    fichier = open("cplx.txt", "w")
    x=[i for i in range(len(points))]
    d=0
    doublons=[]
    simplexes=[{} for i in range(k+1)]
    voisins=[{} for i in range(k+1)]
    for i in range(len(points)):
        simplexes[0][(i,)]=0
        voisins[0][(i,)]=set()
        fichier.write(str(i)+'\n')
    for i,j in itertools.combinations(x,2):
        d= (a_filt(points,[points[i],points[j]]))[1]
        if d<=l and d>=0 :
            simplexes[1][(i,j)]=d
            z=[i,j]
            for r in range(len(z)):
                z[r]=str(z[r])
            fichier.write(' '.join(z)+'\n')
            voisins[0][(i,)].add(j)
            voisins[0][(j,)].add(i)
    for i in simplexes[1].keys():
        voisins[1][i]=(voisins[0][(i)[0],]).intersection(voisins[0][(i)[1],])
    for i in range(2,k+1):
        doublons=[]
        for k,v in simplexes[i-1].items():
            for j in voisins[i-1][k]:
                if set(k+(j,)) not in doublons:
                    d=a_filt(points,[points[m] for m in (k+(j,))])[1]
                    if d<=l and d>=0:
                        doublons+=[set(k+(j,))]
                        simplexes[i][k+(j,)]=d
                        voisins[i][k+(j,)]=(voisins[i-1][k]).intersection(voisins[0][(j,)])
                        z=list(k+(j,))
                        for r in range(len(z)):
                            z[r]=str(z[r])
                        fichier.write(' '.join(z)+'\n')
    return simplexes

def print_a_comp(points,k, l):
    simplexe = a_comp(points,k, l)
    for i in range(k+1):
        for k, v in sorted(simplexe[i].items(), key=lambda x: x[1]):
            print("%s -> [%s]" % (k, v))
    