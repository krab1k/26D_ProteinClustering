from Bio.PDB.PDBParser import PDBParser


def get_aa_residues(pdb, chain):
    """
    pdb: Protein Data Bank file.
    chain: Chain of the PDB file.

    Get the amino acids from a protein.

    returns: List of Biopython PDB Residue objects representing the amino acids
    of the specified protein.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('prot', pdb)
    model = structure[0]
    chain = model[chain]

    return [res for res in chain.get_residues() if res.get_id()[0] == ' ']


def get_aa_name(res):
    """
    res: Biopython PDB Residue object representing an amino acid.

    returns: Name of residue in three letter code + residue number format (e.g. LYS23)
    """
    return res.get_resname() + str(res.get_id()[1])
