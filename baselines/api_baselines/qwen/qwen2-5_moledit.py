# qwen/qwq-32b-preview

import json
from tqdm import tqdm

from openai import OpenAI
ds_client = OpenAI(
    api_key="sk-fbAqopQ5lgukYauf74401607Ca3f4cB0B77f7f5a3b75B032", 
    base_url="https://api.bltcy.ai/v1"
)

def task_specific_system_content(taskname):   
    # if taskname == "add":   
    #     system_content = f"""
    #     You are a chemical assistent. Given the SMILES structural formula of a molecule, help me add a specified functional group and output the improved SMILES sequence of the molecule. Input: Molecule SMILES string, Functional Group Name. Output: Modified Molecule SMILES string.

    #     Your response must contains the step-by-step reasoning, and must be directly parsable JSON format: \n
    #     {{
    #         "molecule_analysis": "[your reasoning] Analyze the functional groups and other components within the molecule",
    #         "function_group_introduce_strategy": "[your reasoning] Determine how and at which site the new group can be most reasonably added",
    #         "feasibility_analysis": "[your reasoning] Assess the chemical viability of the proposed modification",
    #         "output": "Modified Molecule SMILES"
    #     }}
    #     DO NOT output other text except for the answer. If your response includes ```json ```, regenerate it and output ONLY the pure JSON content.
    #     """
    
    # elif taskname == "delete":
    #     system_content = f"""
    #     You are a chemical assistent. Given the SMILES structural formula of a molecule, help me DELETE a specified functional group and output the modified SMILES sequence of the molecule. Input: Molecule SMILES string, Functional Group Name. Output: Modified Molecule SMILES string.

    #     Your response must contains the step-by-step reasoning, and must be directly parsable JSON format: \n
    #     {{
    #         "molecule_analysis": "[your reasoning] Analyze the functional groups and other components within the molecule",
    #         "functional_group_identification": "[your reasoning] Locate the functional group position and analyse",
    #         "delete_strategy": "[your reasoning] Determine how and at which site the functional group can be most reasonably deleted",
    #         "feasibility_analysis": "[your reasoning] Assess the chemical viability of the proposed modification",
    #         "output": "Modified Molecule SMILES"
    #     }}
    #     DO NOT output other text except for the answer. If your response includes ```json ```, regenerate it and output ONLY the pure JSON content.
    #     """
    
    # elif taskname == "sub":
    #     system_content = f"""
    #     You are a chemical assistent. Given the SMILES structural formula of a molecule, help me ADD and DELETE specified functional groups and output the modified SMILES sequence of the molecule. Input: Molecule SMILES string, Functional Group Names. Output: Modified Molecule SMILES string.

    #     Your response must contains the step-by-step reasoning, and must be directly parsable JSON format: \n
    #     {{
    #         "molecule_analysis": "[your reasoning] Analyze the functional groups and other components within the molecule",
    #         "functional_group_identification": "[your reasoning] Locate the functional group position and analyse",
    #         "add_strategy": "[your reasoning] Determine how and at which site the new group can be most reasonably added",
    #         "delete_strategy": "[your reasoning] Determine how and at which site the functional group can be most reasonably deleted",
    #         "feasibility_analysis": "[your reasoning] Assess the chemical viability of the proposed modification",
    #         "output": "Modified Molecule SMILES"
    #     }}
    #     DO NOT output other text except for the answer. If your response includes ```json ```, regenerate it and output ONLY the pure JSON content.
    #     """
    
    if taskname == "add":   
        system_content = f"""
        You are a chemical assistent. Given the SMILES structural formula of a molecule, help me add a specified functional group and output the improved SMILES sequence of the molecule. Input: Molecule SMILES string, Functional Group Name. Output: Modified Molecule SMILES string.

        Your response must be directly parsable JSON format: \n
        {{
            "output": "Modified Molecule SMILES"
        }}
        DO NOT output other text except for the answer. If your response includes ```json ```, regenerate it and output ONLY the pure JSON content.
        """
    
    elif taskname == "delete":
        system_content = f"""
        You are a chemical assistent. Given the SMILES structural formula of a molecule, help me DELETE a specified functional group and output the modified SMILES sequence of the molecule. Input: Molecule SMILES string, Functional Group Name. Output: Modified Molecule SMILES string.

        Your response must be directly parsable JSON format: \n
        {{
            "output": "Modified Molecule SMILES"
        }}
        DO NOT output other text except for the answer. If your response includes ```json ```, regenerate it and output ONLY the pure JSON content.
        """
    
    elif taskname == "sub":
        system_content = f"""
        You are a chemical assistent. Given the SMILES structural formula of a molecule, help me ADD and DELETE specified functional groups and output the modified SMILES sequence of the molecule. Input: Molecule SMILES string, Functional Group Names. Output: Modified Molecule SMILES string.

        Your response must be directly parsable JSON format: \n
        {{
            "output": "Modified Molecule SMILES"
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


def ds_permute(mol_info, user_content, taskname, saving_path):
    ## 这个是作为llm-api在 Molecule-Understanding 任务上的benchmark评测
    ## 包括两个任务 (1). 给定略微突变的两个分子, 让模型比较是否一致 (2). 给定结构一致但是表达不同的分子, 让模型判断两个分子是否一致
    ## 我们通过 task_name 来控制这三个不同的任务切换

    response = ds_client.chat.completions.create(
        model="qwen2.5-32b-instruct",
        messages=[
            { "role": "system", "content": task_specific_system_content(taskname)},
            { "role": "user", "content": user_content},
        ],
        stream=False,
    )
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
    
    
def get_mol_edit_groundtruth_CoT(taskname, modelname):
    ## 生成deepseek在mol-understanding benchmark上的test结果, 包括raw-cot以及json格式的预测输出
    assert taskname in ['add', 'delete', 'sub']
    
    task_info_path = dict(
        add="../../../dataset/mol_edit/1-instruct_to_edit/add.json",
        delete="../../../dataset/mol_edit/1-instruct_to_edit/delete.json",
        sub="../../../dataset/mol_edit/1-instruct_to_edit/sub.json"
    )
    
    task_saving_path = dict(
        add=f"../../api_results/mol_edit/add/cot_results_{modelname}.json",
        delete=f"../../api_results/mol_edit/delete/cot_results_{modelname}.json",
        sub=f"../../api_results/mol_edit/sub/cot_results_{modelname}.json",
    )
    
    file_path = task_info_path[taskname]
    mol_infos = json.load(open(file_path, "r"))
    
    for i in tqdm(range(len(mol_infos)), desc=taskname):
        if taskname == "add":
            smiles, added_group = mol_infos[i]['molecule'], mol_infos[i]['added_group']
            source_content = f"Input Molecule: {smiles}, Functional Group to add: {added_group}." 
            ds_permute(mol_info=mol_infos[i], user_content=source_content, taskname=taskname, saving_path=task_saving_path[taskname])
        
        elif taskname == "delete":
            smiles, removed_group = mol_infos[i]['molecule'], mol_infos[i]['removed_group']
            source_content = f"Input Molecule: {smiles}, Functional Group to delete: {removed_group}." 
            ds_permute(mol_info=mol_infos[i], user_content=source_content, taskname=taskname, saving_path=task_saving_path[taskname])
        
        elif taskname == "sub":
            smiles, added_group, removed_group = mol_infos[i]['molecule'], mol_infos[i]['added_group'], mol_infos[i]['removed_group']
            source_content = f"Input Molecule: {smiles}, Functional Group to delete: {removed_group}, Functional Group to add: {added_group}." 
            ds_permute(mol_info=mol_infos[i], user_content=source_content, taskname=taskname, saving_path=task_saving_path[taskname])
        

if __name__ == "__main__":
    model_name = "qwen25"
    get_mol_edit_groundtruth_CoT(taskname="add", modelname=model_name)
    get_mol_edit_groundtruth_CoT(taskname="delete", modelname=model_name)
    get_mol_edit_groundtruth_CoT(taskname="sub", modelname=model_name)