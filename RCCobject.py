import re
import networkx as nx
from collections import defaultdict, Counter
import RCCdata as rcd
from pymin_graph import *
from pymin_pdb import *


class RCC:
    def __init__(self, pdb, chain):
        residues = get_aa_residues(pdb, chain)
        self.G = build_unweighted_psn(residues, 5.0)
        self.HG = nx.Graph()

        self.number_of_classes = sum(len(x) for x in rcd.setSignatures[3:])
        self.how_many_signatures = Counter()

        self.osisDictString = defaultdict(set)
        self.osisDict = defaultdict(set)
        self.osisDictElements = defaultdict(set)
        self.RCCvector = [0] * self.number_of_classes
        self.RCCvector2 = [0] * self.number_of_classes
        self.metainfo_node = defaultdict(tuple)

        self.createR()

    @staticmethod
    def getSetSignAASeq(cliqueAA):
        """This is a core RCC method"""
        clique = list(map(lambda x: re.sub("\D", "", x[3:]), cliqueAA))
        AAname = dict()
        for aa in cliqueAA:
            AAname[int(re.sub("\D", "", aa[3:]))] = rcd.AA_three_to_one.get(aa[:3], 'X')
        r = sorted(clique, key=lambda item: (int(item.partition(' ')[0])
                                             if item[0].isdigit() else float('inf'), item))
        list_signature = list()
        how_many_consecutive = 1

        for i in range(1, len(clique)):
            if int(r[i]) != int(r[i - 1]) + 1:
                list_signature.append(how_many_consecutive)
                how_many_consecutive = 1
            else:
                how_many_consecutive += 1
        list_signature.append(how_many_consecutive)
        return sorted(list_signature)

    def createR(self):
        cliques = 0
        for q in nx.find_cliques(self.G):
            if (len(q) < 3) or (len(q) > 6):
                continue
            cliques += 1
            tmp_list_sign = self.getSetSignAASeq(q)
            self.how_many_signatures[tuple(tmp_list_sign)] += 1
            L = ','.join(map(str, sorted(tmp_list_sign)))
            self.osisDictString[L].add(','.join(q))
            self.osisDict[L].add(tuple(q))
            map(lambda i: self.osisDictElements[L].add(i), q)

            rcname = hash(tuple(q))
            self.metainfo_node[rcname] = (set(q), tmp_list_sign)
            self.HG.add_node(rcname)
            for hn in self.HG.nodes():
                if self.metainfo_node[hn][0] & self.metainfo_node[rcname][0]:
                    self.HG.add_edge(hn, rcname)

        classindex = 0
        for K in range(3, 7):
            for signa in rcd.setSignatures[K]:
                self.RCCvector[classindex] = self.how_many_signatures[tuple(signa)]
                for n in self.HG.nodes():
                    if self.metainfo_node[n][1] != signa:
                        continue
                    self.RCCvector2[classindex] += self.HG.degree[n]
                classindex += 1
