import os, json, time
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 全局变量（模型和tokenizer）
model = None
tokenizer = None

def task_specific_system_content(taskname):
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

        Your response must contains the step-by-step reasoning, and must be directly parsable JSON format: \n
        {{
            "molecule_A_analysis": "[your reasoning] Structure analysis includes atom count, functional groups, and connectivity of functional groups.",
            "molecule_B_analysis": "[your reasoning] Structure analysis includes atom count, functional groups, and connectivity of functional groups.",
            "atom_count_comparison": "[your reasoning] Check if the types and quantities of atoms in molecule A and molecule B are identical",
            "functional_group_comparison": "[your reasoning] Check if the functional groups and overall structures of molecule A and molecule B are equivalent",
            "output": "Yes / No"
        }}
        DO NOT output other text except for the answer. If your response includes ```json ```, regenerate it and output ONLY the pure JSON content.
        """
    
    elif taskname == "fg_samples":
        system_content = f"""
        [Think short!] You are a chemical assistent. Giving you an Input Molecule and a Fragment name and SMILES, help me count the number of the fragment in the Molecule.
        
        Your response must be directly parsable JSON format: \n
        {{
            "fragment_structure": "fragment structure analysis",
            "matching_analysis": "describe and match the input Molecule with the fragment",
            "count": "Your Answer Number"
        }}
        DO NOT output other text except for the answer. If your response includes ```json ```, regenerate it and output ONLY the pure JSON content. [Think short!]
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
    
    if taskname == "fg_samples":
        smiles = mol_info['smiles']
        fg_name, fg_label = mol_info["fg_name"], mol_info["fg_label"]
        source_content = f"Input Molecule: {smiles}, Fragment SMILES: {fg_label}, Fragment Name: {fg_name}" 
    
    elif taskname == "Murcko_scaffold":
        src_smiles = mol_info['smiles']
        user_content = f"Input Molecule: {src_smiles}."
    
    elif taskname == "ring_count":
        src_smiles, ring_structure = mol_info['smiles'], mol_info['ring']
        user_content = f"Input Molecule: {src_smiles}, Ring Structure: {ring_structure}"
    
    elif taskname == "ring_system_scaffold":
        smiles, ring_scaffold = mol_info['smiles'], mol_info['ring_system_scaffold']
        source_content = f"Input Molecule: {smiles}, Ring System Structure: {ring_scaffold}."
    
    elif taskname == "mutated_list":
        smiles, mutated_smiles = mol_info['smiles'], mol_info['mutated']
        source_content = f"Molecule A: {smiles}, Molecule B: {mutated_smiles}." 
    
    elif taskname == "permutated_list":
        smiles, permutated_smiles = mol_info['smiles'], mol_info['permutated']
        source_content = f"Molecule A: {smiles}, Molecule B: {permutated_smiles}." 

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
        fg_samples="mol_understanding/1-fg_detect/fg_samples.json",
        Murcko_scaffold="mol_understanding/2-frag_detect/Murcko_scaffold.json",
        ring_count="mol_understanding/2-frag_detect/ring_count.json",
        ring_system_scaffold="mol_understanding/2-frag_detect/ring_system_scaffold.json",
        mutated_list="mol_understanding/3-permute_smiles/mutated_list.json",
        permutated_list="mol_understanding/3-permute_smiles/permutated_list.json",
    )
    
    task_saving_path = dict(
        fg_samples=f"fg_samples_results_{save_file_name}.json",
        Murcko_scaffold=f"murcko_results_{save_file_name}.json",
        ring_count=f"ring_count_results_{save_file_name}.json",
        ring_system_scaffold=f"ring_system_results_{save_file_name}.json",
        mutated_list=f"mutated_results_{save_file_name}.json",
        permutated_list=f"permutated_results_{save_file_name}.json",
    )

    taskname_list =['fg_samples', 'Murcko_scaffold', 'ring_count', 'ring_system_scaffold', 'mutated_list', 'permutated_list']
    for taskname in taskname_list:
        und_file_path = task_info_path[taskname]
        und_infos = json.load(open(und_file_path, "r"))

        processed_set = []
        for idx, item in enumerate(tqdm(und_infos, desc=taskname)):
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
