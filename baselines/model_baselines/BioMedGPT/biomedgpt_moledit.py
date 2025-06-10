import os, json, time
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 全局变量（模型和tokenizer）
model = None
tokenizer = None

def task_specific_system_content(taskname):   
    if taskname == "add":   
        system_content = f"""
        You are a chemical assistent. Given the SMILES structural formula of a molecule, help me add a specified functional group and output the improved SMILES sequence of the molecule. Input: Molecule SMILES string, Functional Group Name. Output: Modified Molecule SMILES string.

        Your response must contains the step-by-step reasoning, and must be directly parsable JSON format: \n
        {{
            "molecule_analysis": "[your reasoning] Analyze the functional groups and other components within the molecule",
            "function_group_introduce_strategy": "[your reasoning] Determine how and at which site the new group can be most reasonably added",
            "feasibility_analysis": "[your reasoning] Assess the chemical viability of the proposed modification",
            "output": "Modified Molecule SMILES"
        }}
        DO NOT output other text except for the answer. If your response includes ```json ```, regenerate it and output ONLY the pure JSON content.
        """
    
    elif taskname == "delete":
        system_content = f"""
        You are a chemical assistent. Given the SMILES structural formula of a molecule, help me DELETE a specified functional group and output the modified SMILES sequence of the molecule. Input: Molecule SMILES string, Functional Group Name. Output: Modified Molecule SMILES string.

        Your response must contains the step-by-step reasoning, and must be directly parsable JSON format: \n
        {{
            "molecule_analysis": "[your reasoning] Analyze the functional groups and other components within the molecule",
            "functional_group_identification": "[your reasoning] Locate the functional group position and analyse",
            "delete_strategy": "[your reasoning] Determine how and at which site the functional group can be most reasonably deleted",
            "feasibility_analysis": "[your reasoning] Assess the chemical viability of the proposed modification",
            "output": "Modified Molecule SMILES"
        }}
        DO NOT output other text except for the answer. If your response includes ```json ```, regenerate it and output ONLY the pure JSON content.
        """
    
    elif taskname == "sub":
        system_content = f"""
        You are a chemical assistent. Given the SMILES structural formula of a molecule, help me ADD and DELETE specified functional groups and output the modified SMILES sequence of the molecule. Input: Molecule SMILES string, Functional Group Names. Output: Modified Molecule SMILES string.

        Your response must contains the step-by-step reasoning, and must be directly parsable JSON format: \n
        {{
            "molecule_analysis": "[your reasoning] Analyze the functional groups and other components within the molecule",
            "functional_group_identification": "[your reasoning] Locate the functional group position and analyse",
            "add_strategy": "[your reasoning] Determine how and at which site the new group can be most reasonably added",
            "delete_strategy": "[your reasoning] Determine how and at which site the functional group can be most reasonably deleted",
            "feasibility_analysis": "[your reasoning] Assess the chemical viability of the proposed modification",
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

def load_model(model_name_or_path):
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:3",
    )
    model.eval()

def get_test_data(fold_path):
    with open(os.path.join(fold_path, 'test_data.json'), 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    return test_data

def generate_response_local(sys_prompt, user_prompt):
    global model, tokenizer
    messages = [
        {"role": "system", "content": sys_prompt}, 
        {"role": "user", "content": user_prompt}
    ]

    # 使用Mistral专用的apply_chat_template方法
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8192,  
            do_sample=True,
            temperature=1,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

def process_qa(mol_info, taskname):
    if 'generated_response' in mol_info:
        return mol_info
    
    sys_content = task_specific_system_content(taskname)
    
    if taskname == "add":
        smiles, added_group = mol_info['molecule'], mol_info['added_group']
        source_content = f"Input Molecule: {smiles}, Functional Group to add: {added_group}." 
    
    elif taskname == "delete":
        smiles, removed_group = mol_info['molecule'], mol_info['removed_group']
        source_content = f"Input Molecule: {smiles}, Functional Group to delete: {removed_group}." 
    
    elif taskname == "sub":
        smiles, added_group, removed_group = mol_info['molecule'], mol_info['added_group'], mol_info['removed_group']
        source_content = f"Input Molecule: {smiles}, Functional Group to delete: {removed_group}, Functional Group to add: {added_group}." 

    response = generate_response_local(sys_content, source_content)
    try:
        content_json = json.loads(response)
    except json.JSONDecodeError as e:
        content_json = response
    
    mol_info['json_results'] = content_json
    
    return mol_info

#保存文件
def save_checkpoint(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def main(model_name_or_path, save_file_name):
    fold_path = 'ProtocolGen_Benchmark'
    output_file = f"./SmallResults/{fold_path}_{save_file_name}.json"

    print("Loading model:", model_name_or_path)
    load_model(model_name_or_path)
    
    task_info_path = dict(
        add="mol_edit/1-instruct_to_edit/add.json",
        delete="mol_edit/1-instruct_to_edit/delete.json",
        sub="mol_edit/1-instruct_to_edit/sub.json"
    )
    
    task_saving_path = dict(
        add=f"cot_results_add_{save_file_name}.json",
        delete=f"cot_results_del_{save_file_name}.json",
        sub=f"cot_results_sub_{save_file_name}.json",
    )

    taskname_list =['add', 'delete', 'sub']
    for taskname in taskname_list:
        edit_file_path = task_info_path[taskname]
        edit_infos = json.load(open(edit_file_path, "r"))

        processed_set = []
        for idx, item in enumerate(tqdm(edit_infos, desc=taskname)):
            item = process_qa(item, taskname)
            update_json_file(
                info_dict = item,
                file_name = task_saving_path[taskname]
            )

        # 最后再保存一次确保完整
        save_checkpoint(processed_set, output_file)
        print(f"All data saved to {output_file}")

if __name__ == '__main__':
    for model_path in ['/home/liuyuyang/llzh/hf_models_datasets/BioMistral-7B']:
        main(model_name_or_path=model_path, save_file_name="BioMistral-7B")
