## Gemini评测, 我先在本地跑通一个gemini-api的脚本, 然后让赵赫那边评测
import json
from tqdm import tqdm
from google import genai
from pydantic import BaseModel

your_gemini_api_key = "your_gemini_api_key"

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

class Molopt_Structure(BaseModel):
  Structural_Analysis_of_Source_Molecule: str
  Property_Analysis: str
  Limitation_in_Source_Molecule_for_Property: str
  Optimization_for_Source_Molecule: str
  Final_Target_Molecule: str

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
    
    user_content = f"""
                You are a chemical assistent, Optimize the Source Molecule: {src_smiles} to improve the {prop_descrip_dict[prop]} while following a structured intermediate optimization process. \n\n
                Your response must be directly parsable JSON format:\n
                {{
                    "Structural Analysis of Source Molecule": "",
                    "Property Analysis": "",
                    "Limitation in Source Molecule for Property": ""
                    "Optimization for Source Molecule": "",
                    "Final Target Molecule": "SMILES",
                }}
                DO NOT output other text except for the answer.
             """
    
    client = genai.Client(api_key=your_gemini_api_key)
    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=user_content,
        config={
            'response_mime_type': 'application/json',
            'response_schema': list[Molopt_Structure],
        },
    )
    # Use the response as a JSON string.
    # print(response.text)
        
    update_json_file(
        info_dict = dict(
            src_smiles=src_smiles, prop=prop,
            json_results=response.text,
        ),
        file_name = f"../api_results/deep_mol_opt/{prop}/cot_results_dsr1.json"
    )
    
def get_molopt_cot():
    ## 生成deepseek在benchmark上的test结果, 包括raw-cot以及json格式的预测输出
    prop_list = ['logp', 'solubility', 'qed', 'drd', 'gsk', 'jnk']
    for prop in prop_list:
        mmp_file_path = f"../../dataset/deep_mol_opt/{prop}/final_mmp.json"
        mmp_infos = json.load(open(mmp_file_path, "r"))
        for i in tqdm(range(len(mmp_infos)), desc=prop):
            src_smiles = mmp_infos[i]['src']
            ds_molopt_cot_evaluate(src_smiles=src_smiles, prop=prop)



if __name__ == "__main__":
    get_molopt_cot()
    


