import json
from tqdm import tqdm
from eval.eval_metric import mol_opt_evaluater

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

def eval_molopt_from_list(optimized_prop, gt_list, pred_list, total_number):
    # this_function input: 
    #   gt_list for gt_molecules
    #   pred_list for pred_molecules
    #   total_number: len(gt_molecules)+len(cases that cannot extract SMILES)
    
    prop_dict = dict(logp='logp', solubility='solubility', qed="qed",  drd='drd2', jnk='jnk3', gsk='gsk3b')
    prop_evaluater = mol_opt_evaluater(prop=prop_dict[optimized_prop])
    
    improve_statistic = prop_evaluater.property_improvement(src_mol_list=gt_list, tgt_mol_list=pred_list, total_num=total_number)
    scaffold_hard, scaffold_soft = prop_evaluater.scaffold_consistency(src_mol_list=gt_list, tgt_mol_list=pred_list)
    
    result_dict = dict(
        improvement=improve_statistic,
        scaffold=dict(hard=scaffold_hard/total_number, soft=scaffold_soft/total_number),
    )
    
    return result_dict
    

def evaluate_molopt_score(model_name=None, promptname=None):
    ## evaluating the json_results from llm-api, including qwen, llama, gpt, gemini, ...
    
    prop_dict = dict(logp='logp', solubility='solubility', qed="qed",  drd='drd2', jnk='jnk3', gsk='gsk3b')
    result_final = dict()
    
    for prop in prop_dict.keys():
        print(model_name, prop)
        file_name = f"/cto_labs/lihao/chem_reason/opt_datacollection/distill_models/results/deep_mol_opt/{prop}/cot_results_{model_name}_{promptname}.json"
        pred_results = json.load(open(file_name, "r"))
        tgt_smiles_list, src_smiles_list = list(), list()
        
        invalid_number = 0
        src_smiles_key = 'src'
        
        for pred in pred_results:
            ## 提取 predicted-smiles, 如果生成的是json格式那不需要额外操作, 如果不是, 需要转换成json形式
            if type(pred['pred_json_results']) is str:
                pred_json = tranform_str_to_json(str_input=pred['pred_json_results'])
                # if model_name == 'gemini': pred_json = pred_json[0]
                if pred_json == None or type(pred_json) is str:
                    invalid_number += 1; continue
                else:
                    pred_smiles = pred.get('Final Target Molecule') or pred.get('Final_Target_Molecule') or None
                    if pred_smiles != None:
                        tgt_smiles_list.append(pred_smiles)
                        src_smiles_list.append(pred[src_smiles_key])
                    else: 
                        invalid_number += 1; continue
            else:
                pred_smiles = pred['pred_json_results'].get('Final Target Molecule') or pred['pred_json_results'].get('Final_Target_Molecule') or None
                if pred_smiles != None:
                    tgt_smiles_list.append(pred_smiles)
                    src_smiles_list.append(pred[src_smiles_key])
                else:
                    invalid_number += 1; continue
        
        # double check
        print(len(pred_results), invalid_number, len(src_smiles_list))
        assert len(src_smiles_list) == len(tgt_smiles_list)
        assert len(pred_results) == invalid_number + len(src_smiles_list)
        
        result_dict = eval_molopt_from_list(optimized_prop=prop, gt_list=src_smiles_list, pred_list=tgt_smiles_list, total_number=len(pred_results))
        result_final[prop] = result_dict
        
    
    json.dump(result_final, open(f"/cto_labs/lihao/chem_reason/opt_datacollection/distill_models/results/deep_mol_opt/eval_score_{model_name}_{promptname}.json", "w"), indent=4)
        
if __name__ == "__main__":
    # model_list = ["distill-1.5b", "distill-7b", "distill-14b", "distill-32b"]
    model_list = ["qwen2.5_1.5b", "qwen2.5-7b", "qwen2.5-14b", "qwen2.5-32b"]
    prompt_list = ['raw', 'cot_template', 'cot_groundtruth']
    
    for model_name in model_list:
        for prompt_type in prompt_list:
            evaluate_molopt_score(model_name=model_name, promptname=prompt_type)
    