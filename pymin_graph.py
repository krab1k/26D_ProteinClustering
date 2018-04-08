import networkx as nx
import itertools
import numba
import numpy as np
from pymin_pdb import get_aa_name


def build_unweighted_psn(residues, distance_cutoff):
    """
    residues: Biopython.PDB Residue objects representing the amino acids of a
    protein.

    distance_cutoff: Distance cutoff in Angstroms for edge (interaction) inclusion
    between two nodes (amino acids).

    Builds a protein structure network. That is, a network which nodes are
    the amino acids of a protein and an edge between two nodes represents an
    interaction between the corresponding amino acids. Amino acids that have at
    least one atom pair between them within the specified cutoff distance are assumed
    to interact and an edge between them is included in the network.

    returns: NetworkX Graph object representing the protein structure network.
    """

    @numba.jit(nopython=True, cache=True, nogil=True)
    def distance_sq(coord1, coord2):
        d0 = coord1[0] - coord2[0]
        d1 = coord1[1] - coord2[1]
        d2 = coord1[2] - coord2[2]

        return d0 * d0 + d1 * d1 + d2 * d2

    @numba.jit(nopython=True, cache=True, nogil=True)
    def within_cutoff2(coords1, coords2, distance_cutoff):
        for i in range(coords1.shape[0]):
            for j in range(coords2.shape[0]):
                dist = distance_sq(coords1[i], coords2[j])
                if dist <= distance_cutoff:
                    return True

        return False

    ###########################################################################

    # Create the network and add amino acid residues from the pdb file as nodes.
    network = nx.Graph()
    network.add_nodes_from(residues)

    # Add an edge between every pair of amino acids that has at least one atom pair
    # within the distance cutoff.

    res_coords = [np.array([atom.coord for atom in residue]) for residue in residues]

    distance_cutoff = distance_cutoff ** 2
    for i, j in itertools.combinations(range(len(residues)), 2):
        if within_cutoff2(res_coords[i], res_coords[j], distance_cutoff):
            network.add_edge(get_aa_name(residues[i]), get_aa_name(residues[j]))

    return network
