## DeepSeek-单目标分子优化-评测程序
import json
from tqdm import tqdm

from openai import OpenAI
ds_client = OpenAI(
    api_key="sk-6d2a6aa8e8614801b4b92768195e8600", 
    base_url="https://api.deepseek.com"
)

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

def ds_molopt_cot_evaluate(src_smiles, prop):
    ## 这个是作为llm-api在Benchmark上的评测的, 所以只输入Sourece-Molecule以及对应的优化Property
    prop_descrip_dict = dict(
        drd="DRD2 property (Dopamine D2 Receptor Activity)",
        jnk="JNK3 property (c-Jun N-terminal kinase 3 inhibition)",
        gsk="GSK3-beta property (Glycogen Synthase Kinase 3-beta Inhibition)",
        qed="QED property (Drug-likeness)",
        clint="Hepatic intrinsic clearance (Clint)",
        logp="Distribution coefficient (LogD)",
        solubility="compound's ability to dissolve in water (Solubility)"
    ) 
    user_content = f"Source Molecule: {src_smiles}."
   
    response = ds_client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", 
             "content": f"""
                You are a chemical assistent, Optimize the Source Molecule to improve the {prop_descrip_dict[prop]} while following a structured intermediate optimization process. \n\n
                Your response must be directly parsable JSON format:\n
                {{
                    "Final Target Molecule": "SMILES",
                }}
                DO NOT output other text except for the JSON. If your response includes ```json ```, regenerate it and output ONLY the pure JSON content.  
             """
            },
            {
                "role": "user", 
                "content": user_content
            },
        ],
        # response_format={"type": "json_object"}, 
        stream=False
    )
    reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content
    try:
        content_json = json.loads(content)
    except json.JSONDecodeError as e:
        content_json = content
        
    update_json_file(
        info_dict = dict(
            src_smiles=src_smiles, prop=prop,
            raw_cot=reasoning_content, json_results=content_json,
        ),
        file_name = f"../../api_results/deep_mol_opt/{prop}/cot_results_dsr1.json"
    )
    
def get_molopt_cot():
    ## 生成deepseek在benchmark上的test结果, 包括raw-cot以及json格式的预测输出
    # prop_list = ['logp', 'solubility', 'qed', 'drd', 'gsk', 'jnk']
    prop_list = ['jnk']
    for prop in prop_list:
        mmp_file_path = f"../../../dataset/deep_mol_opt/{prop}/final_mmp.json"
        mmp_infos = json.load(open(mmp_file_path, "r"))
        for i in tqdm(range(len(mmp_infos)), desc=prop):
            src_smiles = mmp_infos[i]['src']
            ds_molopt_cot_evaluate(src_smiles=src_smiles, prop=prop)



if __name__ == "__main__":
    get_molopt_cot()