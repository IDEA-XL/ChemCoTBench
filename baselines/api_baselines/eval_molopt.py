import json
from tqdm import tqdm
from eval_metric import mol_opt_evaluater

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

def evaluate_molopt_score(model_name=None):
    ## 在get_molopt_cot中得到test结果, 我们评测这些test结果
    prop_dict = dict(logp='logp', solubility='solubility', qed="qed",  drd='drd2', jnk='jnk3', gsk='gsk3b')
    # prop_dict = dict(logp='logp', solubility='solubility', qed="qed",  drd='drd2', gsk='gsk3b')
    
    result_final = dict()
    
    for prop in prop_dict.keys():
        print(model_name, prop)
        prop_evaluater = mol_opt_evaluater(prop=prop_dict[prop])
        
        if 'llama' not in model_name:
            file_name = f"../api_results/deep_mol_opt/{prop}/cot_results_{model_name}.json"
        elif model_name == 'llama31-70b': 
            file_name = f"../api_results/deep_mol_opt/{prop}/llama3.1-70b.json"
        elif model_name == 'llama33-nemo-think':
            file_name = f"../api_results/deep_mol_opt/{prop}/llama3.3-nemotron-49B.json"
        elif model_name == 'llama33-nemo':
            file_name = f"../api_results/deep_mol_opt/{prop}/llama3.3-nemotron-49B-non-reason.json"
        
        pred_results = json.load(open(file_name, "r"))
        
        tgt_smiles_list, src_smiles_list = list(), list()
        
        invalid_number = 0
        if model_name != 'gemini': final_target_key = 'Final Target Molecule'
        else: final_target_key = 'Final Target Molecule'
        
        if model_name not in ['biomedgpt', 'biomistral']:
            src_smiles_key =  'src_smiles'
        else: src_smiles_key = "src"
        
        for pred in pred_results:
            ## 提取 predicted-smiles, 如果生成的是json格式那不需要额外操作, 如果不是, 需要转换成json形式
            if type(pred['json_results']) is str:
                pred_json = tranform_str_to_json(str_input=pred['json_results'])
                # if model_name == 'gemini': pred_json = pred_json[0]
                if pred_json == None or type(pred_json) is str:
                    invalid_number += 1
                    continue
                else:
                    if final_target_key in pred_json.keys():
                        tgt_smiles_list.append(pred_json[final_target_key])
                        src_smiles_list.append(pred[src_smiles_key])
            else:
                if final_target_key in pred['json_results'].keys():
                    tgt_smiles_list.append(pred['json_results'][final_target_key])
                    src_smiles_list.append(pred[src_smiles_key])
        
        assert len(src_smiles_list) == len(tgt_smiles_list)
        improve_scores = prop_evaluater.property_improvement(src_mol_list=src_smiles_list, tgt_mol_list=tgt_smiles_list, total_num=len(pred_results))
        scaffold_hard, scaffold_soft = prop_evaluater.scaffold_consistency(src_mol_list=src_smiles_list, tgt_mol_list=tgt_smiles_list)
        result_dict = dict(
            improvement=improve_scores,
            scaffold=dict(hard=scaffold_hard / len(pred_results), soft=scaffold_soft / len(pred_results)),
            invalid_rate=invalid_number/len(pred_results),
        )
        result_final[prop] = result_dict
        
    
    json.dump(result_final, open(f"../api_results/deep_mol_opt//eval_score_{model_name}.json", "w"), indent=4)
        
if __name__ == "__main__":
    ## model_name in ['dsr1', 'o1mini', 'o3mini', 'gpt4o', 'gemini', 'claude3', 'BioMedGPT-LM-7B', 'BioMistral-7B-DARE']
    ## ['dsr1', 'dsv3', 'qwen3', 'qwen3large', 'qwen3largethink', 'qwen3think', 'qwen25', 'claude3', 'llama31-70b', 'llama33-nemo-think']
    
    
    for model_name in ['o1mini']:
        evaluate_molopt_score(model_name=model_name)
        # break
    