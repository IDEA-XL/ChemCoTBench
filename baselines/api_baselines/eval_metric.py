## For Single-Objective Molecule Optimization Benchmark

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import DataStructs
from rdkit.Chem import Descriptors
import rdkit.Chem as Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MurckoScaffoldSmiles

import tdc

def calculate_solubility(smiles):
    ## Calculate aqueous solubility(logS) using RDKit descriptors and a simple linear model.
    try:
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        
        # Calculate relevant descriptors
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        h_bond_donors = Descriptors.NumHDonors(mol)
        h_bond_acceptors = Descriptors.NumHAcceptors(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        # Simple linear model (based on published QSPR models)
        logS = 0.16 - 0.63*logp - 0.0062*mw + 0.066*h_bond_donors - 0.074*h_bond_acceptors
        return logS
    
    except Exception as e:
        print(f"Error calculating solubility: {e}")
        return 0.0

def compute_statistics(numbers, prop):
    if numbers == []:
        return {
            "mean": 0,
            "variance": 0,
            "min": 0,
            "max": 0,
            "success_rate": 0, # success opt that increase the property
            "best_rate": 0, # rate of best property mol-opt
        }

    easy_thres, hard_thres = 0.5, 0.3
    threshold_dict = {'gsk3b':hard_thres, 'qed':hard_thres, 'drd2':hard_thres, 'jnk3':hard_thres, 'logp':easy_thres, 'solubility':easy_thres}
    
    n = len(numbers)
    mean = sum(numbers) / n
    # Calculate variance (using population variance: 1/N * sum((x_i - mean)^2))
    variance = sum((x - mean) ** 2 for x in numbers) / n
    min_val = min(numbers)
    max_val = max(numbers)
    
    success_rate = sum(1 for itm in numbers if itm > 0) / len(numbers)
    best_rate = sum(1 for itm in numbers if itm >= threshold_dict[prop]) / len(numbers)
    
    return {
        "mean": mean,
        "variance": variance,
        "min": min_val,
        "max": max_val,
        "success_rate": success_rate, # success opt that increase the property
        "best_rate": best_rate, # rate of best property mol-opt
    }

class mol_opt_evaluater():
    def __init__(self, prop=None, ) -> None:
        ## prop: item in ['gsk3b', 'qed', 'drd2', 'jnk3']
        self.prop = prop
        if prop in ['gsk3b', 'qed', 'drd2', 'jnk3', 'logp']:
            self.property_oracle = tdc.Oracle(name=prop)
        elif prop == 'solubility':
            self.property_oracle = calculate_solubility
    
    def property_improvement(self, src_mol_list, tgt_mol_list, total_num):
        ## evaluate the property improvement after the mol-opt
        assert len(src_mol_list) == len(tgt_mol_list)
        prop_improve_list = [
            self.property_oracle(tgt_mol_list[i])-self.property_oracle(src_mol_list[i]) for i in range(len(tgt_mol_list))
        ]
        prop_improve_list = prop_improve_list + [0.0]*(total_num - len(src_mol_list))
        statistic = compute_statistics(prop_improve_list, self.prop)
        return statistic
    
    def scaffold_consistency(self, src_mol_list, tgt_mol_list):
        ## evaluate the scaffold consistency before&after mol-opt, consistency includes: same or contain
        assert len(src_mol_list) == len(tgt_mol_list)
        
        count_same = 0
        scaffold_score = list()
        
        for i in range(len(tgt_mol_list)):
            src_smiles, tgt_smiles = src_mol_list[i], tgt_mol_list[i]
            try:
                src_mol, tgt_mol = Chem.MolFromSmiles(src_smiles), Chem.MolFromSmiles(tgt_smiles)
            except:
                continue
            
            if src_mol == None or tgt_mol == None:
                scaffold_score.append(0.0)
                continue
            
            opt_smiles = [src_smiles, tgt_smiles]
            murcko_scaffold_list = [MurckoScaffoldSmiles(smiles) for smiles in opt_smiles]
            
            if len(set(murcko_scaffold_list)) == 1:
                scaffold_score.append(1.0)
                count_same += 1
            else:
                ## Morgan Fingerprint for scaffold similarity
                murcko_scaffold_mol_list = [Chem.MolFromSmiles(murcko_scaffold_list[0]), Chem.MolFromSmiles(murcko_scaffold_list[1])]
                mcs = rdFMCS.FindMCS(murcko_scaffold_mol_list)
                mcs_mol = Chem.MolFromSmarts(mcs.smartsString) if mcs.numAtoms > 0 else None
                
                if mcs_mol:
                    # 计算基于指纹的Tanimoto相似度
                    fp1 = AllChem.GetMorganFingerprintAsBitVect(murcko_scaffold_mol_list[0], 2, nBits=1024)
                    fp2 = AllChem.GetMorganFingerprintAsBitVect(murcko_scaffold_mol_list[1], 2, nBits=1024)
                    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                else:
                    similarity = 0.0
                
                scaffold_score.append(similarity)  
        
        if len(tgt_mol_list) == 0:
            return 0.0, 0.0
        
        return count_same, sum(scaffold_score)
    
    def CoT_explanation(self, src_mol_list, tgt_mol_list, cot_list):
        ## For explanation, CoT format and correctness
        pass