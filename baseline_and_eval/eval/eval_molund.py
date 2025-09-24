
import json
from eval.eval_metric import mol_opt_evaluater

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

def eval_molund_from_list(gt_list, pred_list, total_number, task):
    # this_function input: 
    #   gt_list for gt_molecules
    #   pred_list for pred_molecules
    #   total_number: len(gt_molecules)+len(cases that cannot extract SMILES)
    score = None
    if task in ["ring_system", "ring_system_scaffold", "permutated"]: # "ring_system_scaffold", "ring_system" as the same
        count = sum(1 for item in pred_list if str(item).lower() == "yes")
        if len(pred_list) == 0: score = None
        else: score = count / len(pred_list)
    elif task == "mutated":
        count = sum(1 for item in pred_list if str(item).lower() == "no")
        if len(pred_list) == 0: score = None
        else: score = count / len(pred_list)
    elif task == 'equivalence': # "equivalence" = "mutated" + "permutated"
        count = 0
        for i in range(len(pred_list)):
            if str(pred_list[i]).lower() == str(gt_list[i]).lower():
                count += 1
        if len(pred_list) == 0: score = None
    elif task in ["ring_count", "fg_samples", "fg_count"]:
        assert len(gt_list) == len(pred_list)
        if len(gt_list) == 0: score = None
        else: score = sum([abs(int(pred_list[i])-int(gt_list[i])) for i in range(len(pred_list))]) / len(gt_list)
    elif task in ["murcko", "Murcko_scaffold"]: # "murcko" "Murcko_scaffold" are the same
        assert len(gt_list) == len(pred_list)
        prop_evaluater = mol_opt_evaluater(prop='qed')
        scaffold_hard, scaffold_soft = prop_evaluater.scaffold_consistency(src_mol_list=gt_list, tgt_mol_list=pred_list)
        if len(gt_list) == 0: score = None
        else: score = scaffold_soft / len(pred_list)
    
    my_dict = {
        "score": score,
        f"{task}-valid-rate": len(pred_list)/total_number
    }
    
    return my_dict
            