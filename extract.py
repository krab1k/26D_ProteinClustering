import argparse
import os
from RCCobject import RCC
import numpy as np
from multiprocessing import Pool
import tqdm

def create_rcc(data):
    pdb, chain = data
    return RCC(pdb, chain).RCCvector


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clustering of protein chains')
    parser.add_argument('pdb_directory')
    parser.add_argument('list_of_chains')
    parser.add_argument('output_file')

    args = parser.parse_args()

    ids = []
    with open(args.list_of_chains) as chains_f:
        for line in chains_f:
            pdb_id, chain_id = line.strip().split(':')
            ids.append((pdb_id, chain_id))

    with Pool(6) as pool:
        data = [(os.path.join(args.pdb_directory, f'pdb{pdb_id.lower()}.ent'), chain_id) for pdb_id, chain_id in ids]
        rccs = list(tqdm.tqdm(pool.imap(create_rcc, data), total=len(ids)))

    arr = np.array(rccs)
    np.savetxt(args.output_file, arr, fmt='%4d', delimiter=',')
