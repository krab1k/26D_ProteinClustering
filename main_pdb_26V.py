import Bio
from Bio.PDB import PDBList
import RCCobject as rcu
pdbs, chains= [], []
f = open('pdb_ids.txt','r')# path: correct_chains file
#print f.read()
data = f.readlines()
for line in data:
    i=0
    print (line)
    for l in line.strip().split(':'):
        if i==0:
            pdbs.append(l)
            i+=1
        else:
            chains.append(l)
f.close()
for i in range(len(pdbs)):
    print(pdbs[i])
    print(chains[i])
    pdb=str('pdbs\\'+pdbs[i]+'.pdb')#path: PDB_Files
    rccs= rcu.RCC(pdb,chains[i])
    print ','.join(map(str,rccs.RCCvector))# 26 descriptors of PDB:Chain
    fh = open('file26Vectors.txt', 'a+')
    donnees=pdbs[i]+chains[i]+','+','.join(map(str,rccs.RCCvector))+'\n'
    fh.write(donnees)
    fh.flush()
    fh.close
f = open('file26Vectors.txt','r')
print f.readlines()
f.close


