import requests, os, json
from tqdm import tqdm
from openai import OpenAI
import time
from multiprocessing import Pool, cpu_count
import itertools,random
random.seed(42)

def get_test_data(fold_path):
    test_data=json.load(open(fold_path+'/test_data.json','r'))
    return test_data


def generate_response(user_prompt, model_name, max_retries=5, initial_delay=1):
    client = OpenAI(
        ###xiugai###
        #柏拉图
        api_key="sk-baqvjodgissagchegjuzsbfpfqkysgqlrjfvykxqjzkzzjxv", 
        base_url="https://api.bltcy.ai/v1"
    )
    
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            full_response_content=''
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                stream=True,
                max_tokens=8192
            )
            #return response.choices[0].message.content
            for chunk in response:
                # Each chunk might contain a delta with content
                # Check if delta and content exist in the chunk
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response_content += content
            return full_response_content
            
        except Exception as e:
            last_exception = e
            print(f"Attempt {attempt} failed, {e}")
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                time.sleep(delay)
    
    raise Exception(f"All {max_retries} attempts failed") from last_exception

def process_qa(qa, model_name):
    if 'generated_response' in qa:
        return qa

    user_prompt = f"""{qa['system_prompt']}
{qa['instruction']}
Format requirements:
- Each step must be on a separate line.

{qa['input']}"""

    response = generate_response(user_prompt, model_name=model_name)
    qa['generated_response'] = response
    return qa

def process_item(args):
    item, model_name = args
    if all('generated_response' in qa for qa in item['QA']):
        return item
    
    new_qa = [process_qa(qa, model_name) for qa in item['QA']]
    item['QA'] = new_qa
    return item

def save_checkpoint(data, filename):
    with open(filename, 'w',encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def main(model_name,save_file_name):
    ###xiugai###
    fold_path = 'ProtocolGen_Benchmark'
    output_file = f"./SmallResults/{fold_path}_{save_file_name}.json"
    ###xiugai###
    
    print("Loading model", model_name)
    
    # Load test data
    if os.path.exists(output_file):
        print("Loading from checkpoint")
        with open(output_file, 'r',encoding='utf-8') as f:
            test_set = json.load(f)
    else:
        print("Loading test data from", fold_path)
        test_set = get_test_data(fold_path=fold_path)
    
    # Prepare arguments for multiprocessing
    num_processes = min(cpu_count(),16)  # Limit to 8 processes to avoid rate limiting
    
    # Process in batches with progress tracking
    batch_size = 100
    processed_set = []
    
    with Pool(processes=num_processes) as pool:
        for i in tqdm(range(0, len(test_set), batch_size), desc="Processing batches"):
            batch = test_set[i:i + batch_size]
            processed_batch = list(tqdm(pool.imap(process_item, zip(batch, itertools.repeat(model_name))), 
                                      total=len(batch), 
                                      desc=f"Batch {i//batch_size + 1}", 
                                      leave=False))
            
            processed_set.extend(processed_batch)
            
            # Save checkpoint after each batch
            if i % batch_size == 0:  # Save every 1 batches
                save_checkpoint(processed_set + test_set[len(processed_set):], output_file)
                print(f"Checkpoint saved at batch {i//batch_size + 1}")
    
    # Final save
    save_checkpoint(processed_set, output_file)
    print(f"All data saved to {output_file}")

if __name__ == '__main__':
    for model_name in ['qwq-32b']:
        main(model_name=model_name,save_file_name=model_name)