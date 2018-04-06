import networkx as nx
import sys, os
from operator import itemgetter
from heapq import nlargest
from collections import defaultdict, Counter
from networkx.algorithms import isomorphism
import RCCdata as rcd
import getfirstchain as rcu
import re

class RCCbase(object):
        def __init__(self,pdb,chain,autochain,chain_segments):
                """
                This constructor creates only a Residue Interaction Graph of a given protein chain.
                chain_segments can be given as a list of pirs indicating start and end position of a segment in chain.
                If chain_segments is an empty list, all chain is considered.
                """
                self.pdb = pdb
                self.chain = chain

                if autochain:
                        chain = rcu.getFirstChain(pdb)
                #os.system('java -cp /home/rcc/libs/ PDBparser ' + pdb + ' ' + chain  + ' > tmpgraph');
                #Calculates distances between all atoms in a structure.
                #Filters the distances if a cutoff value is given (default 5.0A).
                #os.system('C:\\Python27\\python make_RIG.py ' + pdb + ' ' + chain  + ' 5.0 > tmpgraph');
                os.system('python make_RIG.py ' + pdb + ' ' + chain  + ' 5.0 > tmpgraph');
                os.system('rm -rf output*')
                fin = open('tmpgraph',"r")

                self.G = nx.Graph()
                self.HG = nx.Graph()
                for linea in fin:
                        if linea.strip():
                                a, b = map(str,linea.strip().split())
                                a_insegment, b_insegment = True, True

                                if chain_segments!=[]:
                                        a_insegment, b_insegment = False, False
                                        for segment in chain_segments:
                                                if  (int(a[3:]) in xrange(segment[0],segment[1]+1)): a_insegment = True
                                                if  (int(b[3:]) in xrange(segment[0],segment[1]+1)): b_insegment = True

                                if a_insegment and b_insegment:
                                        self.G.add_edge(a,b)
                if len(self.G.nodes()) == 0:
                        raise Exception("Wrong graph construction")

class RCC(RCCbase):
        def __init__(self,pdb,chain,autochain=False,chain_segments=[]):
                super(RCC,self).__init__(pdb,chain,autochain,chain_segments)
                self.number_of_classes = sum(map(lambda x:len(x),rcd.setSignatures[3:]))
                self.how_many_signatures = Counter()


                self.osisDictString = defaultdict(set)
                self.osisDict = defaultdict(set)
                self.osisDictElements = defaultdict(set) 
                self.RCCvector = [0]*self.number_of_classes
                self.RCCvector2 = [0]*self.number_of_classes
                self.metainfo_node = defaultdict(tuple) 

                self.createR()

        def getSetSignAASeq(self,cliqueAA,gapped=True,gapchar='_'):  
                """This is a core RCC method"""
                clique = map(lambda x:re.sub("\D", "", x[3:]),cliqueAA)
                AAname = dict()
                for aa in cliqueAA:
                        AAname[int(re.sub("\D","",aa[3:]))] = rcd.AA_three_to_one.get(aa[:3],'X')
                r = sorted(clique, key=lambda item: (int(item.partition(' ')[0])
                                        if item[0].isdigit() else float('inf'), item))
                list_signature = list()
                how_many_consecutive = 1
                secuencia = str()       
                secuencia += AAname[int(r[0])]
                for i in range(1,len(clique)):
                         if(int(r[i])!=int(r[i-1])+1):
                                if gapped: secuencia += gapchar
                                list_signature.append(how_many_consecutive)
                                how_many_consecutive = 1
                         else:
                                how_many_consecutive+=1
                         secuencia += AAname[int(r[i])]
                list_signature.append(how_many_consecutive)
                return dict(list_signature=sorted(list_signature),secuencia=secuencia) 


        def createR(self): 
                clases = set() 
                cliques = 0
                for q in  nx.find_cliques(self.G):
                        if (len(q) <3) or (len(q)>6) : continue
                        cliques += 1
                        tmp_list_sign = self.getSetSignAASeq(q)['list_signature']
                        self.how_many_signatures[tuple(tmp_list_sign)] += 1     
                        L = ','.join(map(lambda(x):str(x),sorted(tmp_list_sign)))
                        self.osisDictString[L].add(','.join(q))
                        self.osisDict[L].add(tuple(q))
                        map(lambda(i):self.osisDictElements[L].add(i),q)

                        rcname =  hash(tuple(q))
                        self.metainfo_node[rcname] = (set(q),tmp_list_sign)
                        self.HG.add_node(rcname)
                        for hn in self.HG.nodes():
                                if self.metainfo_node[hn][0] & self.metainfo_node[rcname][0]:
                                        self.HG.add_edge(hn,rcname)

                classindex = 0
                for K in xrange(3,7):
                        for signa in rcd.setSignatures[K]:
                                self.RCCvector[classindex] = self.how_many_signatures[tuple(signa)]
                                for n in self.HG.nodes():
                                        if self.metainfo_node[n][1] != signa: continue
                                        self.RCCvector2[classindex] += self.HG.degree(n)
                                classindex += 1


#rccs= RCC('12as.pdb','B')
#print '\n rccs.RCCvector : ',rccs.RCCvector
#print ','.join(map(str,rccs.RCCvector))
#print '\n teste----------------------------\n'
#fh = open('file2.txt', 'a+')
#donnees='12as'+'A'+','+','.join(map(str,rccs.RCCvector))+'\n'
#fh.write(donnees)
#fh.flush()
#fh.close
#f = open('file2.txt','r')
#print f.readlines()
#f.close
