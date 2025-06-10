import os, json, time
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 全局变量（模型和tokenizer）
model = None
tokenizer = None

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

def generate_response_local(user_prompt):
    global model, tokenizer
    messages = [
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

def process_qa(qa):
    if 'generated_response' in qa:
        return qa

    user_prompt = f"""
    {qa['system_prompt']}
    {qa['instruction']}
    Format requirements:
    - Each step must be on a separate line.

    {qa['input']}"""

    response = generate_response_local(user_prompt)
    qa['generated_response'] = response
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

    # 加载数据
    if os.path.exists(output_file):
        print("Loading from checkpoint:", output_file)
        with open(output_file, 'r', encoding='utf-8') as f:
            test_set = json.load(f)
    else:
        print("Loading test data from:", fold_path)
        test_set = get_test_data(fold_path=fold_path)

    processed_set = []
    for idx, item in enumerate(tqdm(test_set, desc="Processing items one by one")):
        #跳过已处理的数据
        if 'generated_response' in item:
            processed_set.append(item)
            continue

        item = process_qa(item)
        processed_set.append(item)

        # 每处理完一个样本就保存一次
        save_checkpoint(processed_set + test_set[len(processed_set):], output_file)

    # 最后再保存一次确保完整
    save_checkpoint(processed_set, output_file)
    print(f"All data saved to {output_file}")

if __name__ == '__main__':
    for model_path in ['/home/liuyuyang/llzh/hf_models_datasets/BioMistral-7B']:
        main(model_name_or_path=model_path, save_file_name="BioMistral-7B")
