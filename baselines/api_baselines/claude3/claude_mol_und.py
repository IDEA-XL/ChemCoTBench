
import os
import json
from tqdm import tqdm

from openai import OpenAI
ds_client = OpenAI(
    api_key="sk-fbAqopQ5lgukYauf74401607Ca3f4cB0B77f7f5a3b75B032", 
    base_url="https://api.bltcy.ai/v1"
)

def task_specific_system_content(taskname):
    # if taskname == "Murcko_scaffold":        
    #     system_content = f"""
    #     [Keep reasoning under 10 sentences!!]
    #     You are a chemistry expert performing Murcko scaffold extraction. Input: a molecule's SMILES string. Output: the Murcko scaffold in SMILES format.

    #     Definition: The Murcko scaffold is obtained by removing all side chains, functional groups, and exocyclic modifications, leaving only the ring systems and connecting bonds.

    #     Your response must be directly parsable JSON format: \n
    #     {{
    #         "Output Scaffold": "SMILES"
    #     }}
    #     DO NOT output other text except for the answer. [Keep reasoning under 10 sentences!!]
    #     """
        
    # elif taskname == "ring_count":
    #     system_content = f"""
    #     You are a chemistry expert. Help me count the number of ring structure in the Input Molecule.
    #     Your response must be directly parsable JSON format: \n
    #     {{
    #         "count": "your number"
    #     }}
    #     DO NOT output other text except for the answer.
    #     """
    # elif taskname == "ring_system_scaffold":
    #     system_content = f"""
    #     [Think short!] You are a chemical assistent. Please Determine whether the ring_system_scaffold is in the Molecule. Input: a molecule's SMILES string, a Ring System Scaffold. Output: yes / no.

    #     Definition: The ring system scaffold consists of one or more cyclic (ring-shaped) molecular structures

    #     Your response must be directly parsable JSON format: \n
    #     {{
    #         "output": "Yes / No"
    #     }}
    #     DO NOT output other text except for the answer. If your response includes ```json ```, regenerate it and output ONLY the pure JSON content. [Think short!]
    #     """
    
    # elif taskname == "mutated_list" or taskname == "permutated_list":
    #     system_content = f"""
    #     You are a chemical assistent. Given two molecule SMILES, Please Determine whether these two Molecules are the same. Input: Molecule A SMILES string, Molecule B SMILES string. Output: yes / no. 

    #     Your response must contains the step-by-step reasoning, and must be directly parsable JSON format: \n
    #     {{
    #         "molecule_A_analysis": "[your reasoning] Structure analysis includes atom count, functional groups, and connectivity of functional groups.",
    #         "molecule_B_analysis": "[your reasoning] Structure analysis includes atom count, functional groups, and connectivity of functional groups.",
    #         "atom_count_comparison": "[your reasoning] Check if the types and quantities of atoms in molecule A and molecule B are identical",
    #         "functional_group_comparison": "[your reasoning] Check if the functional groups and overall structures of molecule A and molecule B are equivalent",
    #         "output": "Yes / No"
    #     }}
    #     DO NOT output other text except for the answer. If your response includes ```json ```, regenerate it and output ONLY the pure JSON content.
    #     """
    
    # elif taskname == "fg_samples":
    #     system_content = f"""
    #     [Think short!] You are a chemical assistent. Giving you an Input Molecule and a Fragment name and SMILES, help me count the number of the fragment in the Molecule.
        
    #     Your response must be directly parsable JSON format: \n
    #     {{
    #         "fragment_structure": "fragment structure analysis",
    #         "matching_analysis": "describe and match the input Molecule with the fragment",
    #         "count": "Your Answer Number"
    #     }}
    #     DO NOT output other text except for the answer. If your response includes ```json ```, regenerate it and output ONLY the pure JSON content. [Think short!]
    #     """
    
    if taskname == "Murcko_scaffold":        
        system_content = f"""
        [Keep reasoning under 10 sentences!!]
        You are a chemistry expert performing Murcko scaffold extraction. Input: a molecule's SMILES string. Output: the Murcko scaffold in SMILES format.

        Definition: The Murcko scaffold is obtained by removing all side chains, functional groups, and exocyclic modifications, leaving only the ring systems and connecting bonds.

        Your response must be directly parsable JSON format: \n
        {{
            "Output Scaffold": "SMILES"
        }}
        DO NOT output other text except for the answer. [Keep reasoning under 10 sentences!!]
        """
        
    elif taskname == "ring_count":
        system_content = f"""
        You are a chemistry expert. Help me count the number of ring structure in the Input Molecule.
        Your response must be directly parsable JSON format: \n
        {{
            "count": "your number"
        }}
        DO NOT output other text except for the answer.
        """
    elif taskname == "ring_system_scaffold":
        system_content = f"""
        [Think short!] You are a chemical assistent. Please Determine whether the ring_system_scaffold is in the Molecule. Input: a molecule's SMILES string, a Ring System Scaffold. Output: yes / no.

        Definition: The ring system scaffold consists of one or more cyclic (ring-shaped) molecular structures

        Your response must be directly parsable JSON format: \n
        {{
            "output": "Yes / No"
        }}
        DO NOT output other text except for the answer. If your response includes ```json ```, regenerate it and output ONLY the pure JSON content. [Think short!]
        """
    
    elif taskname == "mutated_list" or taskname == "permutated_list":
        system_content = f"""
        You are a chemical assistent. Given two molecule SMILES, Please Determine whether these two Molecules are the same. Input: Molecule A SMILES string, Molecule B SMILES string. Output: yes / no. 

        Your response must be directly parsable JSON format: \n
        {{
            "output": "Yes / No"
        }}
        DO NOT output other text except for the answer. If your response includes ```json ```, regenerate it and output ONLY the pure JSON content.
        """
    
    elif taskname == "fg_samples":
        system_content = f"""
        [Think short!] You are a chemical assistent. Giving you an Input Molecule and a Fragment name and SMILES, help me count the number of the fragment in the Molecule.
        
        Your response must be directly parsable JSON format: \n
        {{
            "count": "Your Answer Number"
        }}
        DO NOT output other text except for the answer. If your response includes ```json ```, regenerate it and output ONLY the pure JSON content.
        """
    
    return system_content
    

def update_json_file(info_dict, file_name='data.json'):
    try:
        with open(file_name, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # 如果文件不存在或文件为空，则创建一个新的空列表
        data = []

    # 将新的info_dict添加到数据列表中
    data.append(info_dict)

    # 将更新后的数据写回文件
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)


def ds_frag_detect(mol_info, user_content, taskname, saving_path):
    ## 这个是作为llm-api在 Molecule-Understanding 任务上的benchmark评测
    ## 包括三个任务 (1). 输入分子smiles得到scaffold (2). 输入分子smiles得到环数量的计数 (3). 输入分子smiles得到ring_system的scaffold
    ## 我们通过 task_name 来控制这三个不同的任务切换

    response = ds_client.chat.completions.create(
        model="claude-3-7-sonnet-20250219",
        messages=[
            { "role": "system", "content": task_specific_system_content(taskname)},
            { "role": "user", "content": user_content},
        ],
        stream=False
    )
    # reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content
    
    try:
        content_json = json.loads(content)
    except json.JSONDecodeError as e:
        content_json = content
    
    mol_info['task'] = taskname
    mol_info['json_results'] = content_json
    update_json_file(
        info_dict = mol_info,
        file_name=saving_path
    )
    
    
def predict_mol_understanding(taskname, modelname):
    ## 生成deepseek在mol-understanding benchmark上的test结果, 包括raw-cot以及json格式的预测输出
    assert taskname in ['fg_category', 'fg_samples', 'Murcko_scaffold', 'ring_count', 'ring_system_scaffold', 'mutated_list', 'permutated_list']
    
    task_info_path = dict(
        fg_category="../../../dataset/mol_understanding/1-fg_detect/fg_category.json",
        fg_samples="../../../dataset/mol_understanding/1-fg_detect/fg_samples.json",
        Murcko_scaffold="../../../dataset/mol_understanding/2-frag_detect/Murcko_scaffold.json",
        ring_count="../../../dataset/mol_understanding/2-frag_detect/ring_count.json",
        ring_system_scaffold="../../../dataset/mol_understanding/2-frag_detect/ring_system_scaffold.json",
        mutated_list="../../../dataset/mol_understanding/3-permute_smiles/mutated_list.json",
        permutated_list="../../../dataset/mol_understanding/3-permute_smiles/permutated_list.json",
    )
    
    task_saving_path = dict(
        fg_samples=f"../../api_results/mol_understanding/fg_samples/cot_results_{modelname}.json",
        Murcko_scaffold=f"../../api_results/mol_understanding/frag_detect_murcko/cot_results_{modelname}.json",
        ring_count=f"../../api_results/mol_understanding/frag_detect_ring_count/cot_results_{modelname}.json",
        ring_system_scaffold=f"../../api_results/mol_understanding/frag_detect_ring_system/cot_results_{modelname}.json",
        mutated_list=f"../../api_results/mol_understanding/mutated/cot_results_{modelname}.json",
        permutated_list=f"../../api_results/mol_understanding/permutated/cot_results_{modelname}.json",
    )
    
    file_path = task_info_path[taskname]
    mol_infos = json.load(open(file_path, "r"))
    
    if os.path.exists(task_saving_path[taskname]):
        done_infos = json.load(open(task_saving_path[taskname], "r"))
        mol_infos = mol_infos[len(done_infos): ]
    
    for i in tqdm(range(len(mol_infos)), desc=taskname):
        if taskname == "fg_samples":
            smiles = mol_infos[i]['smiles']
            fg_name, fg_label = mol_infos[i]["fg_name"], mol_infos[i]["fg_label"]
            source_content = f"Input Molecule: {smiles}, Fragment SMILES: {fg_label}, Fragment Name: {fg_name}" 
            ds_frag_detect(mol_info=mol_infos[i], user_content=source_content, taskname=taskname, saving_path=task_saving_path[taskname])
        
        elif taskname == "Murcko_scaffold":
            src_smiles = mol_infos[i]['smiles']
            user_content = f"Input Molecule: {src_smiles}."
            ds_frag_detect(mol_info=mol_infos[i], user_content=user_content, taskname=taskname, saving_path=task_saving_path[taskname])
        
        elif taskname == "ring_count":
            src_smiles, ring_structure = mol_infos[i]['smiles'], mol_infos[i]['ring']
            user_content = f"Input Molecule: {src_smiles}, Ring Structure: {ring_structure}"
            ds_frag_detect(mol_info=mol_infos[i], user_content=user_content, taskname=taskname, saving_path=task_saving_path[taskname])
        
        elif taskname == "ring_system_scaffold":
            smiles, ring_scaffold = mol_infos[i]['smiles'], mol_infos[i]['ring_system_scaffold']
            source_content = f"Input Molecule: {smiles}, "
            source_content += f"Ring System Structure: {ring_scaffold}"
            
            ds_frag_detect(mol_info=mol_infos[i], user_content=source_content, taskname=taskname, saving_path=task_saving_path[taskname])
        
        elif taskname == "mutated_list":
            smiles, mutated_smiles = mol_infos[i]['smiles'], mol_infos[i]['mutated']
            source_content = f"Molecule A: {smiles}, Molecule B: {mutated_smiles}." 
            ds_frag_detect(mol_info=mol_infos[i], user_content=source_content, taskname=taskname, saving_path=task_saving_path[taskname])
        
        elif taskname == "permutated_list":
            smiles, permutated_smiles = mol_infos[i]['smiles'], mol_infos[i]['permutated']
            source_content = f"Molecule A: {smiles}, Molecule B: {permutated_smiles}." 
            ds_frag_detect(mol_info=mol_infos[i], user_content=source_content, taskname=taskname, saving_path=task_saving_path[taskname])



if __name__ == "__main__":
    model_name = "claude3"
    predict_mol_understanding(taskname="Murcko_scaffold", modelname=model_name)
    predict_mol_understanding(taskname="ring_count", modelname=model_name)
    predict_mol_understanding(taskname="ring_system_scaffold", modelname=model_name)
    
    predict_mol_understanding(taskname="mutated_list", modelname=model_name)
    predict_mol_understanding(taskname="permutated_list", modelname=model_name)
    
    predict_mol_understanding(taskname="fg_samples", modelname=model_name)
    