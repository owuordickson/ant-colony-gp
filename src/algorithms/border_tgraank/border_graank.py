# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:31:16 2015

@author: Olivier + modif MJL+MR 140316

Modified by Dickson Owuor Sat Feb 23 18:17:35 2019

"""
import csv
import numpy as np
import gc
import sys
import ntpath

from mbdll_border import *


def Trad(fileName):
    temp=[]
    with open(fileName, 'rU') as f:
        dialect = csv.Sniffer().sniff(f.read(1024), delimiters=";,' '\t")
        f.seek(0)
        reader = csv.reader(f, dialect)
        temp = list(reader)
        f.close()
    #print(temp)
    if temp[0][0].replace('.','',1).isdigit() or temp[0][0].isdigit():
        return [[float(temp[j][i]) for j in range(len(temp))] for i in range(len(temp[0]))]
    else:
        if temp[0][1].replace('.','',1).isdigit() or temp[0][1].isdigit():
            return [[float(temp[j][i]) for j in range(len(temp))] for i in range(1,len(temp[0]))]
        else:
            title = []
            for i in range(len(temp[0])):
                sub = (str(i + 1) + ' : ' + temp[0][i])
                title.append(sub)
            return title, [[float(temp[j][i]) for j in range(1, len(temp))] for i in range(len(temp[0]))]


def GraankInit(T,eq=False):
    res=[]
    n=len(T[0])
    #print T
    for i in range(len(T)):
        npl=str(i+1)+'+'
        nm=str(i+1)+'-'
        tempp=np.zeros((n,n),dtype= 'bool')
        tempm=np.zeros((n,n),dtype= 'bool')
        #print i
        for j in range(n):
            for k in range(j+1,n):
                if T[i][j]>T[i][k]:
                    tempp[j][k]=1
                    tempm[k][j]=1
                else:
                    if T[i][j]<T[i][k]:
                        #print (j,k)
                        tempm[j][k]=1
                        tempp[k][j]=1
                    else:
                        if eq:
                            tempm[j][k]=1
                            tempp[k][j]=1
                            tempp[j][k]=1
                            tempm[k][j]=1
        res.append((set([npl]),tempp))
        res.append((set([nm]),tempm))
    return res


def SetMax(R):
    i=0
    k=0
    test=0
    Cb=R
    while(i<len(Cb)-1):
            test=0
            k=i+1
            while(k<len(Cb)):
                if(Cb[i].issuperset(Cb[k]) or Cb[i]==Cb[k]):
                    del Cb[k]
                else:
                    if Cb[i].issubset(Cb[k]):
                        del Cb[i]
                        test=1
                        break
                k+=1
            if test==1:
                continue
            i+=1
    return Cb


def inv(s):
    i=len(s)-1
    if s[i]=='+':
        return s[0:i]+'-'
    else:
        return s[0:i]+'+'


def APRIORIgen(R,a,n):
    res=[]
    test=1
    temp=set()
    temp2=set()
    #print"a"
    I=[]
    if(len(R)<2):
        return []
    Ck=[x[0] for x in R]
    #print"b"
    for i in range(len(R)-1):
        #print"c"
        #print len(R)
        for j in range(i+1,len(R)):
            temp=R[i][0]|R[j][0]
            invtemp={inv(x) for x in temp}
            #print invtemp
            #print"d"+str(j)
            if ((len(temp)==len(R[0][0])+1) and (not (I!=[] and temp in I)) and (not (I!=[] and invtemp in I))):
                test=1
                #print "e"
                for k in temp:
                    temp2=temp-set([k])
                    invtemp2={inv(x) for x in temp2}
                    if not temp2 in Ck and not invtemp2 in Ck:
                        test=0
                        break
                if test==1:
                    m=R[i][1]*R[j][1]
                    t=float(np.sum(m))/float(n*(n-1.0)/2.0)
                    if t >a:
                        res.append((temp,m))
                I.append(temp)
                gc.collect()

    #print "z"
    return res


def Graank(D_in,a,eq=False):
    title = D_in[0]
    T = D_in[1]
    res = []
    res2 = []
    temp = 0
    n = len(T[0])
    G = GraankInit(T,eq)
    #print G
    for i in G:
        temp = float(np.sum(i[1]))/float(n*(n-1.0)/2.0)
        if temp < a:
            G.remove(i)
#        else:
#            res.append(i[0])
    while G!=[]:
        G=APRIORIgen(G,a,n)
        #print G
        i=0
        while i<len(G) and G!=[]:
            temp=float(np.sum(G[i][1]))/float(n*(n-1.0)/2.0)
            #print temp
            if temp<a:
                del G[i]
            else:
                #print i
                z=0
                while z <(len(res)-1):
                    if res[z].issubset(G[i][0]):
                        del res[z]
                        del res2[z]
                    else:
                        z=z+1
                res.append(G[i][0])
                res2.append(temp)
                i+=1
    return title, res, res2


def fuse(L):
    Res=L[0][:][:4000]
    for j in range(len(L[0])):
        for i in range(1,len(L)):
            Res[j]=Res[j]+L[i][j][:4000]
    return Res


def fuseTrad(L):
    temp=[]
    for i in L:
        temp.append(Trad(i))
    return fuse(temp)


def getSupp(T,s,eq=False):
    n=len(T[0])
    res=0
    for i in range(len(T[0])):
        for j in range(i+1,len(T[0])):
            temp=1
            tempinv=1
            for k in s:
                x=int(k[0:(len(s)-1)])-1
                if(k[len(s)-1]=='+'):
                    if(T[x][i]>T[x][j]):
                        tempinv=0
                    else:
                        if(T[x][i]<T[x][j]):
                            temp=0
                else:
                    if(T[x][i]<T[x][j]):
                        tempinv=0
                    else:
                        if(T[x][i]>T[x][j]):
                            temp=0
                if(T[x][i]==T[x][j] and not eq):
                    temp=0
                    tempinv=0
            res=res+temp+tempinv
    return float(res)/float(n*(n-1.0)/2.0)


#def main(filename1,supmin1,eq=False):
#    D1,S1=Graank(Trad(filename1),supmin1,eq)
#    print('D1 : '+filename1)
#    for i in range(len(D1)):
#        print(str(D1[i])+' : '+str(S1[i]))
#main('FluTopicData-testsansdate-blank.csv',0.5,False)
#main('ndvi_file.csv',0.5,False)

# --------------------- CODE FOR EMERGING PATTERNS -------------------------------------------


def get_maximal_items(init_list):
    # comb = list((zip(init_list, tlag_list)))
    max_items = gen_set(tuple(init_list))
    temp = list(max_items)

    for item_i in max_items:
        for item_j in max_items:
            if set(item_i).issubset(set(item_j)) and set(item_i) != (set(item_j)):
                try:
                    if item_i in temp:
                        temp.remove(item_i)
                except:
                    continue
    return temp


# ------------------------- main method --------------------------------


def algorithm_gradual(file_name, min_sup):
    title, D1, S1=Graank(Trad(file_name), min_sup, False)
    #print(str(D1))
    for line in title:
        print(line)
    print('<h5>Pattern : Support</h5>')
    if D1:
        for i in range(len(D1)):
            supp = "%.2f" % S1[i]
            print(str(tuple(D1[i])) + ' : ' + str(supp) + "<br>")
        sys.stdout.flush()
    else:
        print("<h5>Oops! no gradual patterns found</h5>")
        sys.stdout.flush()


def algorithm_ep_gradual(file_path_1, file_path_2, min_sup):
    try:

        # 1. get Gradual patterns for dataset 1 and 2
        title_1, gp_list_1, S1 = Graank(Trad(file_path_1), min_sup, False)
        title_2, gp_list_2, S2 = Graank(Trad(file_path_2), min_sup, False)

        # 2. check if data-sets have matching columns
        if title_1 == title_2:
            if gp_list_1 and gp_list_2:
                # 3. get maximal item-sets
                freq_pattern_1 = get_maximal_items(gp_list_1)
                freq_pattern_2 = get_maximal_items(gp_list_2)

                # 4. get emerging gradual patterns
                ep = mbdll_border(tuple(freq_pattern_1), tuple(freq_pattern_2))
                if not ep:
                    print("Oops! no relevant emerging pattern was found")
                    print("-------------------------------------------")
                else:
                    for line in title_1:
                        print(line)

                    file_1 = ntpath.basename(file_path_1)
                    file_2 = ntpath.basename(file_path_2)
                    print(str(file_2) + " opposing " + str(file_1))
                    print("---------------------------------------")
                    print(str(ep))
            else:
                print("Oops! no frequent patterns were found")
                print("---------------------------------------")
        else:
            print("Data-sets do not match")
            print("-----------------------")

    except Exception as error:
        print(error)


algorithm_ep_gradual('FluTopicData.csv', 'FluTopicData.csv', .1)

