# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 17:48:52 2022

@author: Yorgo Chamoun
"""

import numpy as np
import math as m
import itertools
import random
import time

def dist(x,y):
    return m.sqrt(sum([(x[i]-y[i])**2 for i in range(len(x))]))

def tab_aretes(G):
    t_rep=np.sort((np.array(G))[:,2])
    t=[t_rep[0]]
    for i in range(len(t_rep)-1):
        if t_rep[i]!=t_rep[i+1]:
            t+=[t_rep[i+1]]
    e={}
    for i in t:
        e[i]=[]
    for i in G:
        e[i[2]]+=[(i[0],i[1])]
    return t,e
        
def mat(G):
    n=max([(max([G[i][0], G[i][1]])) for i in range(len(G))])
    M=[[0 for i in range(n+1)] for j in range(n+1)]
    return M,n

def ajouter(M,u,v):
    M[u][v]=1
    M[v][u]=1
    
def supprimer(M,u,v):
    M[u][v]=0
    M[v][u]=0
    
def commun(M,u,v,n):
    commun=[u,v]
    for j in range(n+1):
        if M[u][j]!=0 and M[v][j]!=0:
            commun+=[j]
    return commun

def domination(M,u,v,n):
    communs=commun(M,u,v,n)
    if len(communs)>2:
        k=2
        fini=0
        while k<len(communs):
            l=0
            fini=0
            while l<len(communs) and fini==0:
                if M[communs[k]][communs[l]]==0 and k!=l:
                    fini=1
                l+=1
            if fini==0:
                return True
            k+=1
    return False

def simp(G):
    t,e=tab_aretes(G)
    m=len(t)
    M,n=mat(G)
    fichier = open("graphe.txt", "w")
    for i in range(m):
        for u,v in e[t[i]]:
            ajouter(M,u,v)  #il faut tout ajouter avant!
        for u,v in e[t[i]]:
            if domination(M,u,v,n):
                supprimer(M,u,v)
                if i!=m-1:
                    e[t[i+1]]+=[(u,v)]
            else:
                fichier.write(str(u)+" "+str(v)+" "+str(t[i])+"\n")
    return M

def complet(n):
    d=1
    g=[]
    x=[]
    x=[i for i in range(n)]
    fichier = open("complet.txt", "w")
    for i in itertools.combinations(x,2):
        z=list(i)
        x=z.copy()
        for r in range(2):
            z[r]=str(z[r])
        x+=[d]
        fichier.write(' '.join(z)+" "+'1'+'\n')
        g+=[x]
    fichier.close()
    return(g)

def polygon(n):
    g=[]
    x=[]
    points=[[m.sin(i*2*m.pi/n),m.cos(i*2*m.pi/n)] for i in range(n)]
    x=[i for i in range(n)]
    fichier = open("polygon.txt", "w")
    for i in itertools.combinations(x,2):
        z=list(i)
        x=z.copy()
        d=round(dist(points[z[0]],points[z[1]]),5)
        for r in range(2):
            z[r]=str(z[r])
        x+=[d]
        fichier.write(' '.join(z)+" "+str(d)+'\n')
        g+=[x]
    fichier.close()
    return(g)
    
def alea(n):
    g=[]
    x=[]
    points=[[random.random(),random.random()] for i in range(n)]
    x=[i for i in range(n)]
    fichier = open("alea.txt", "w")
    for i in itertools.combinations(x,2):
        z=list(i)
        x=z.copy()
        d=round(dist(points[z[0]],points[z[1]]),5)
        for r in range(2):
            z[r]=str(z[r])
        x+=[d]
        fichier.write(' '.join(z)+" "+str(d)+'\n')
        g+=[x]
    fichier.close()
    return(g)
    
def tab_aretes_opt(G):
    t_rep=np.sort((np.array(G))[:,2])
    t=[t_rep[0]]
    for i in range(len(t_rep)-1):
        if t_rep[i]!=t_rep[i+1]:
            t+=[t_rep[i+1]]
    e={}
    for i in t:
        e[i]=[[],[]]
    for i in G:
        e[i[2]][0]+=[(i[0],i[1])]
    return t,e
 
def simp_opt(G):
    t,e=tab_aretes_opt(G)
    m=len(t)
    M,n=mat(G)
    dom=0
    fichier = open("graphe.txt", "w")
    for i in range(m):
        for u,v in e[t[i]][1]:
            ajouter(M,u,v)
        for u,v in e[t[i]][0]:
            ajouter(M,u,v)  #il faut tout ajouter avant!
        dom=1
        for u,v in e[t[i]][0]:
            if domination(M,u,v,n):
                supprimer(M,u,v)
                if i!=m-1:
                    e[t[i+1]][1]+=[(u,v)]
            else:
                dom=0
                fichier.write(str(u)+" "+str(v)+" "+str(t[i])+"\n")
        if dom==0:
            for u,v in e[t[i]][1]:
                if domination(M,u,v,n):
                    supprimer(M,u,v)
                    if i!=m-1:
                        e[t[i+1]][1]+=[(u,v)]
                else:
                    fichier.write(str(u)+" "+str(v)+" "+str(t[i])+"\n")
        else:
            for u,v in e[t[i]][1]:
                    supprimer(M,u,v)
            if i!=m-1:
                e[t[i+1]][1]+=e[t[i]][1]
    return M

def simp_time(G,N):
    dt1=0
    dt2=0
    for i in range(N):
        t1=time.time()
        simp(G)
        dt1+=time.time()-t1
        t2=time.time()
        simp_opt(G)
        dt2+=time.time()-t2
    return dt1/N,dt2/N  