import json
import os
from rdkit import Chem
from rdkit.Chem import AllChem
def read_txt(path):
    # parse lines
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        return lines
    
def write_txt(path, data):
    with open(path, 'w') as f:
        for line in data:
            f.write(line + '\n')

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def is_valid_smiles(smiles, strict:bool=True):
    if strict:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
    else:
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return False
        except:
            return False
    return True

def is_same_rxn_smiles(rxn_smiles1, rxn_smiles2):
    """Check if two reaction SMILES are the same"""
    def merge_reagents_to_reactants(rxn_smiles):
        reactants, reagents, products = rxn_smiles.split('>')
        reactants = reactants.split('.')
        reagents = reagents.split('.')
        products = products.split('.')
        if reagents != ['']:
            reactants = reactants + reagents
        return '.'.join(reactants) + '>>' + '.'.join(products)

    
    rxn_smiles1 = merge_reagents_to_reactants(rxn_smiles1)
    rxn_smiles2 = merge_reagents_to_reactants(rxn_smiles2)

    # Remove atom maps first
    rxn_smiles1 = rxn_remove_atom_map(rxn_smiles1)
    rxn_smiles2 = rxn_remove_atom_map(rxn_smiles2)

    # Split into reactants and products
    reactants1, products1 = rxn_smiles1.split('>>')
    reactants2, products2 = rxn_smiles2.split('>>')
    
    # Split into individual molecules and sort
    reactants1 = sorted(reactants1.split('.'))
    reactants2 = sorted(reactants2.split('.'))
    products1 = sorted(products1.split('.'))
    products2 = sorted(products2.split('.'))
    
    # Compare reactants and products
    return reactants1 == reactants2 and products1 == products2

def safe_remove_hs(mol):
    try:
        # 先处理芳香性
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        # 再移除氢原子
        return Chem.RemoveHs(mol)
    except:
        # 如果失败，尝试替代方案
        mol = Chem.Mol(mol)
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return Chem.RemoveHs(mol)
    
def get_smiles_without_map(mol, remove_Hs:bool=False):
    """生成去除原子映射号的规范SMILES"""
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol, sanitize=False)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {mol}")
    if remove_Hs:
        mol = safe_remove_hs(mol)
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')
    return Chem.MolToSmiles(mol, canonical=True)

def rxn_remove_atom_map(rxn_w_atom_map:str, remove_Hs:bool=False):
    """
    Remove atom map from reaction SMILES.

    NOTE: rxn_w_atom_map is like this:
    C1CC(C(C(C(C1)O)O)O)O.O.O>>C1CC(C(C(C(C1)O)O)O)O.O.O
    """
    reactants, products = rxn_w_atom_map.split('>>')
    reactants = reactants.split('.')
    products = products.split('.')
    # check each reactant and product is valid
    for reactant in reactants:
        if Chem.MolFromSmiles(reactant, sanitize=False) is None:
            raise ValueError(f"Invalid reactant: {reactant}")
    for product in products:
        if Chem.MolFromSmiles(product, sanitize=False) is None:
            raise ValueError(f"Invalid product: {product}")
    reactants = [get_smiles_without_map(reactant) for reactant in reactants]
    products = [get_smiles_without_map(product) for product in products]
    return '.'.join(reactants) + '>>' + '.'.join(products)

def rxn_remove_atom_map_w_reagents(rxn_w_atom_map:str, remove_Hs:bool=False):
    """
    Remove atom map from reaction SMILES.
    
    NOTE: rxn_w_atom_map is like this:
    C1CC(C(C(C(C1)O)O)O)O.O.O>C1CC(C(C(C(C1)O)O)O)O.O.O>C1CC(C(C(C(C1)O)O)O)O.O.O
    """
    reactants, reagents, products = rxn_w_atom_map.split('>')
    reactants = reactants.split('.')
    products = products.split('.')
    reagents = reagents.split('.')
    # check each reactant and product is valid
    for reactant in reactants:
        if Chem.MolFromSmiles(reactant, sanitize=False) is None:
            raise ValueError(f"Invalid reactant: {reactant}")
    for product in products:
        if Chem.MolFromSmiles(product, sanitize=False) is None:
            raise ValueError(f"Invalid product: {product}")
    for reagent in reagents:
        if Chem.MolFromSmiles(reagent, sanitize=False) is None:
            raise ValueError(f"Invalid reagent: {reagent}")
    reactants = [get_smiles_without_map(reactant) for reactant in reactants]
    products = [get_smiles_without_map(product) for product in products]
    reagents = [get_smiles_without_map(reagent) for reagent in reagents]
    return '.'.join(reactants) + '>' + '.'.join(reagents) + '>' + '.'.join(products)

def parse_flowER_mech_step(mech:str):
    """Parse flowER mechanism"""
    # {rxn_smi}|{reaction_class}|reaction_condition}|{des}|{rxn_id}
    rxn_smi, rxn_cls, rxn_condition, des, rxn_id = mech.split('|')
    return {
        "rxn_smi": rxn_smi,
        "rxn_cls": rxn_cls,
        "rxn_condition": rxn_condition,
        "des": des,
        "rxn_id": rxn_id
    }

def extract_agents_from_rxn(reaction_smiles, remove_Hs:bool=False, separate_agents:bool=False):
    """
    Extract agents from reaction SMILES.

    reaction_smiles: str
    remove_Hs: bool, whether to remove Hs
    separate_agents: bool, whether to separate reagents, if True, return 'reactants>reagents>products', otherwise return 'reactants.reagents>>products'
    """
    # 解析反应
    rxn = AllChem.ReactionFromSmarts(reaction_smiles)
    reactants = rxn.GetReactants()
    products = rxn.GetProducts()

    # 收集反应物和产物的映射信息
    reactant_info = []
    product_info = []

    # 处理反应物
    for mol in reactants:
        maps = set()
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                maps.add(int(atom.GetProp('molAtomMapNumber')))
        smiles = get_smiles_without_map(mol)
        reactant_info.append( (frozenset(maps), smiles, mol) )

    # 处理产物
    for mol in products:
        maps = set()
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                maps.add(int(atom.GetProp('molAtomMapNumber')))
        smiles = get_smiles_without_map(mol)
        product_info.append( (frozenset(maps), smiles, mol) )

    # 识别未变化的试剂
    agents = []
    used_reactant_indices = set()
    used_product_indices = set()

    # 遍历所有反应物-产物对
    for r_idx, (r_maps, r_smiles, r_mol) in enumerate(reactant_info):
        for p_idx, (p_maps, p_smiles, p_mol) in enumerate(product_info):
            if r_maps == p_maps and r_smiles == p_smiles:
                # 记录匹配的分子
                agents.append(Chem.MolToSmiles(r_mol, True))
                used_reactant_indices.add(r_idx)
                used_product_indices.add(p_idx)
                break

    # 构建新的反应方程式
    new_rxn = AllChem.ChemicalReaction()

    # 添加未使用的反应物
    for idx, mol in enumerate(reactants):
        if idx not in used_reactant_indices:
            new_rxn.AddReactantTemplate(mol)

    # 添加未使用的产物
    for idx, mol in enumerate(products):
        if idx not in used_product_indices:
            new_rxn.AddProductTemplate(mol)

    # 生成结果
    rxn_smi = AllChem.ReactionToSmiles(new_rxn)
    if remove_Hs:
        agents = [get_smiles_without_map(agent, True) for agent in agents]
        rxn_smi = rxn_remove_atom_map(rxn_smi, True)

    if not separate_agents:
        return agents, rxn_smi
    else:
        reactants, products = rxn_smi.split('>>')
        rxn_smi = reactants + '>' + ".".join(agents) + '>' + products
        return agents, rxn_smi
         
def smiles_to_smarts(smiles):
    """Convert SMILES to SMARTS pattern"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Invalid SMILES
    return Chem.MolToSmarts(mol)

def count_fragments(smiles, frag_smiles):
    """Count occurrences of a fragment in a molecule"""
    # Convert fragment SMILES to SMARTS pattern
    frag_pattern = Chem.MolFromSmiles(frag_smiles)
    if frag_pattern is None:
        return 0  # Invalid fragment SMILES
    
    # Convert the input SMILES to a molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0  # Invalid SMILES
    
    # Find all matches of the fragment pattern
    matches = mol.GetSubstructMatches(frag_pattern)
    return len(matches)