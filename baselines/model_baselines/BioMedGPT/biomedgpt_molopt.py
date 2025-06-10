import os, json, time
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 全局变量（模型和tokenizer）
model = None
tokenizer = None

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

def process_qa(qa, prop):
    if 'generated_response' in qa:
        return qa
    
    prop_descrip_dict = dict(
        drd="DRD2 property (Dopamine D2 Receptor Activity)",
        jnk="JNK3 property (c-Jun N-terminal kinase 3 inhibition)",
        gsk="GSK3-beta property (Glycogen Synthase Kinase 3-beta Inhibition)",
        qed="QED property (Drug-likeness)",
        clint="Hepatic intrinsic clearance (Clint)",
        logp="Distribution coefficient (LogD)",
        solubility="compound's ability to dissolve in water (Solubility)"
    ) 
    
    sys_content = f"""
                You are a chemical assistent, Optimize the Source Molecule to improve the {prop_descrip_dict[prop]} while following a structured intermediate optimization process. \n\n
                Always output in strict, raw JSON format. Do NOT include any Markdown code block wrappers (e.g., ```json ``` or ```). Your response must be directly parsable JSON format:\n
                {{
                    "Structural Analysis of Source Molecule": "",
                    "Property Analysis": "",
                    "Limitation in Source Molecule for Property": ""
                    "Optimization for Source Molecule": "",
                    "Final Target Molecule": "SMILES",
                }}
                DO NOT output other text except for the answer. If your response includes ```json ```, regenerate it and output ONLY the pure JSON content.  
             """
    
    user_content = f"Source Molecule: {qa['src']}."

    response = generate_response_local(sys_content, user_content)
    try:
        content_json = json.loads(response)
    except json.JSONDecodeError as e:
        content_json = response
    
    qa['json_results'] = content_json
    
    return qa

#保存文件
def save_checkpoint(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def main(model_name_or_path, save_file_name):
    fold_path = 'ProtocolGen_Benchmark'
    output_file = f"./SmallResults/{fold_path}_{save_file_name}.json"

    print("Loading model:", model_name_or_path)
    load_model(model_name_or_path)

    prop_list = ['logp', 'solubility', 'qed', 'drd', 'gsk', 'jnk']
    for prop in prop_list:
        mmp_file_path = f"{prop}/final_mmp.json"
        mmp_infos = json.load(open(mmp_file_path, "r"))

        processed_set = []
        for idx, item in enumerate(tqdm(mmp_infos, desc="Processing items one by one")):

            item = process_qa(item, prop)
            update_json_file(
                info_dict = item,
                file_name = f"{prop}/cot_results_{save_file_name}.json"
            )


        # 最后再保存一次确保完整
        save_checkpoint(processed_set, output_file)
        print(f"All data saved to {output_file}")

if __name__ == '__main__':
    for model_path in ['/home/liuyuyang/llzh/hf_models_datasets/BioMistral-7B']:
        main(model_name_or_path=model_path, save_file_name="BioMistral-7B")
