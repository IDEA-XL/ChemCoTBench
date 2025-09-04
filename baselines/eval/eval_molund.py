
import json
from tqdm import tqdm

from eval_metric import mol_opt_evaluater

def check_string_type(s):
    try:
        int(s)
        return "int"
    except ValueError:
        try:
            float(s)
            return "float"
        except ValueError:
            return "string"

def tranform_str_to_json(str_input):
    ## 假如LLM输出的是类似json的字符串, 我需要设定一个逻辑, 把字符串重新转换成json
    ## o1-mini的感觉, 是要移除字符串里面的\n，并且把所有的\"都改成 "
    if "</think>\n\n" in str_input:
        str_input = str_input.split("</think>\n\n")[-1]
    
    if "```json\n" in str_input:
        str_input = str_input.split("```json\n")[1]
        str_input = str_input.replace("\n```", '')
    
    unescaped_str = str_input.replace('\n    ', '').replace('\n', '').replace('\"', '"')
    try:
        json_obj = json.loads(unescaped_str)
        return json_obj
    except json.JSONDecodeError as e:
        return None
    
def evaluate_molund_score(model_name=None, promptname=None):
    ## 在get_molopt_cot中得到test结果, 我们评测这些test结果
    task_dict = dict(
        fg_samples="fg_samples", murcko='frag_detect_murcko', ring_count='frag_detect_ring_count',
        ring_system='frag_detect_ring_system'
    )
    pred_key_dict = dict(
        fg_samples="count", murcko='Output Scaffold', ring_count='count',
        ring_system='output', mutated='output', permutated='output'
    )
    gt_key_dict = dict(
        fg_samples="fg_num", murcko='largest_scaffold', ring_count='count',
        ring_system='', mutated='', permutated=''
    )
    prop_evaluater = mol_opt_evaluater(prop='qed')
    
    result_dict = dict()
    
    for task in task_dict.keys():
        print(model_name, task)
        
        if 'llama' not in model_name:
            file_name = f"results/mol_understanding/{task_dict[task]}/cot_results_{model_name}_{promptname}.json"
            
        pred_results = json.load(open(file_name, "r"))
        
        invalid_number = 0
        
        pred_list, gt_list = list(), list()
        for pred in pred_results:
            ## 提取 predicted-smiles, 如果生成的是json格式那不需要额外操作, 如果不是, 需要转换成json形式    
            if type(pred['json_results']) is str:
                pred_json = tranform_str_to_json(str_input=pred['json_results'])
                # if model_name == 'gemini': pred_json = pred_json[0]
                if pred_json == None:
                    invalid_number += 1
                    continue
                else:
                    if pred_key_dict[task] not in pred_json.keys():
                        invalid_number += 1; continue
                    if pred_json[pred_key_dict[task]] == "": 
                        invalid_number += 1; continue
                    if task in ["ring_count", "fg_samples"]:
                        if check_string_type(pred_json[pred_key_dict[task]]) == "string":
                            invalid_number += 1; continue;
                    pred_list.append(pred_json[pred_key_dict[task]])
                    if gt_key_dict[task] != "":
                        gt_list.append(pred[gt_key_dict[task]])
            else:
                if pred_key_dict[task] not in pred['json_results'].keys():
                    invalid_number += 1; continue
                if pred['json_results'][pred_key_dict[task]] == "": 
                    invalid_number += 1; continue
                if task in ["ring_count", "fg_samples"]:
                    if check_string_type(pred['json_results'][pred_key_dict[task]]) == "string":
                            invalid_number += 1; continue;
                pred_list.append(pred['json_results'][pred_key_dict[task]])
                if gt_key_dict[task] != "":
                    gt_list.append(pred[gt_key_dict[task]])
        
        # print(f"invalid = {invalid_number} / {len(pred_results)}")
        if task in ["ring_system", "permutated"]:
            count = sum(1 for item in pred_list if str(item).lower() == "yes")
            if len(pred_list) == 0: score = None
            else: score = count / len(pred_list)
        elif task == "mutated":
            count = sum(1 for item in pred_list if str(item).lower() == "no")
            if len(pred_list) == 0: score = None
            else: score = count / len(pred_list)
        elif task in ["ring_count", "fg_samples"]:
            assert len(gt_list) == len(pred_list)
            if len(gt_list) == 0: score = None
            else: score = sum([abs(int(pred_list[i])-int(gt_list[i])) for i in range(len(pred_list))]) / len(gt_list)
        elif task == "murcko":
            assert len(gt_list) == len(pred_list)
            scaffold_hard, scaffold_soft = prop_evaluater.scaffold_consistency(src_mol_list=gt_list, tgt_mol_list=pred_list)
            if len(gt_list) == 0: score = None
            else: score = scaffold_soft / len(pred_list)
        
        result_dict[task] = score
        result_dict[f"{task}-valid-rate"] = (len(pred_results)-invalid_number) / len(pred_results) 
        print(model_name, task, score)
    
    json.dump(result_dict, open(f"results/mol_understanding/eval_score_{model_name}_{promptname}.json", "w"), indent=4)


if __name__ == "__main__":
    model_list = ["distill-1.5b", "distill-7b", "distill-14b", "distill-32b"]
    # model_list = ["qwen2.5_1.5b", "qwen2.5-7b", "qwen2.5-14b", "qwen2.5-32b"]
    prompt_list = ['raw', 'cot_template', 'cot_groundtruth']
    
    for model_name in model_list:
        for prompt_type in prompt_list:
            evaluate_molund_score(model_name=model_name, promptname=prompt_type)
            