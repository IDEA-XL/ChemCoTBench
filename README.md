<div align="center">
  <img src="https://github.com/HowardLi1984/ChemCoTBench/blob/main/figures/chemcotbench-intro.png?raw=true" alt="ChemCoTBench Logo" width="900"/>
</div>


# Beyond Chemical QA: Evaluating LLM's Chemical Reasoning with Modular Chemical Operations

[![ArXiv](https://img.shields.io/badge/ArXiv-paper-B31B1B.svg?logo=arXiv&logoColor=Red)](https://arxiv.org/abs/2505.21318)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Dataset-FFD210.svg?logo=HuggingFace&logoColor=black)](https://huggingface.co/datasets/OpenMol/ChemCoTBench)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Homepage](https://img.shields.io/badge/Homepage-brightgreen.svg)](https://howardli1984.github.io/ChemCoTBench.github.io/)
[![LeaderBoard](https://img.shields.io/badge/Leaderboard-welcome-blue.svg)](https://howardli1984.github.io/ChemCoTBench.github.io/)

**ChemCoTBench is the first large-scale benchmark for step-wise reasoning on complex chemical problems, specifically designed for large language models (LLMs). It moves beyond simple QA to encompass a comprehensive suite of tasks critical for chemical comprehension.**

---

<div align="center">
  <img src="https://github.com/HowardLi1984/ChemCoTBench/blob/main/figures/chemcot-distribution.png" alt="ChemCoTBench Logo" width="1000"/>
</div>

ChemCoTBench features:
* 📚 **Large-scale Data:** Built upon **2M original chemical molecule samples**, yielding nearly **20K high-quality chain-of-thoughts samples**.
* 🎯 **Comprehensive Tasks:** A suite of **four core tasks** challenging LLMs on different facets of chemical tasks:

    * Molecule SMILES-level Understanding
    * Molecule Murcko-Scaffold Understanding
    * Molecule Functional Group Counting
    * Molecule Editing (Add, Delete, Substitute)
    * Molecule Optimization for Physicochemical Properties (QED, LogP, Solubility)
    * Molecule Optimization for Protein Activation (DRD-2, JNK-3, GSK-3beta)
    * Retrosynthesis Prediction
    * Forward Major-Product Prediction
    * Forward By-Product Prediction
    * Reaction Condition Prediction
    * Reaction Mechanism Prediction
* 🔬 **Standardized Evaluation:** A robust framework combining standard NLP metrics with novel domain-specific measures for accurate performance quantification.
  
---
## 🚀 Motivation

Despite recent advances in LLM reasoning capabilities, chemistry, a discipline fundamental to areas like drug discovery and materials science, still lacks a benchmark that assesses whether these improvements extend to its complex, domain-specific problem-solving needs. While several benchmarks have been proposed for LLMs in chemistry, they primarily focus on domain-specific question answering, which suffers from several key limitations:

* **Lack of Structured, Stepwise Reasoning and Real-World Relevance:** Current evaluations often reduce chemistry assessment to factual recall (e.g., naming compounds or reactions), neglecting the need for operational reasoning akin to arithmetic or coding. Unlike mathematical problems, where solutions demand explicit, verifiable steps, chemistry QA tasks fail to simulate how experts decompose challenges. For instance, they don't capture the process of iteratively refining a molecule’s substructure to optimize properties, considering crucial real-world factors like synthesizability or toxicity, or deducing reaction mechanisms through intermediate transformations. This gap means we're not fully evaluating the analytical depth required in real-world chemistry. Therefore, evaluations must shift from these textbook-like problems to challenges that better reflect practical applications.

* **Ambiguous Skill Attribution in Hybrid Evaluations:** Existing benchmarks often conflate reasoning, knowledge recall, and numerical computation into single "exam-style" metrics—for instance, asking LLMs to calculate reaction yields while simultaneously recalling reagent properties. This obscures whether strong performance stems from structured reasoning (e.g., analyzing reaction pathways) or memorized facts (e.g., solvent boiling points). Such ambiguity hinders targeted model improvement and misaligns evaluations with downstream tasks like drug discovery, where success depends on modular reasoning (e.g., decoupling molecular design from synthesizability checks) rather than monolithic problem-solving.

To address these limitations, we introduce ChemCoTBench, a **step-by-step, application-oriented, and high-quality benchmark** for evaluating LLM reasoning in chemical applications. A core innovation of ChemCoTBench is its formulation of complex chemical tasks, specifically targeting molecular modeling and design, into explicit sequences of verifiable modular chemical operations on SMILES structures (e.g., substructure addition, deletion, or substitution). This approach allows for a granular assessment of an LLM's ability to execute and chain together fundamental chemical transformations. The benchmark features progressively challenging tasks, spanning from basic molecular understanding and editing to property-guided structure optimization and complex multi-molecule chemical reactions. High-quality evaluation is ensured through a dual validation process combining LLM judgment with expert review from 13 chemists.

---
## 📊 Huggingface Dataset & Benchmark

To visualize the Dataset Samples, and the baseline Leaderboard, please check the Leaderboard-page and huggingface repo.

* **Leaderboard-Page:** [https://howardli1984.github.io/ChemCoTBench.github.io/](https://howardli1984.github.io/ChemCoTBench.github.io/)
* **ChemCoTBench:** [https://huggingface.co/datasets/OpenMol/ChemCoTBench](https://huggingface.co/datasets/OpenMol/ChemCoTBench)
* **Large-scale ChemCoTDataset:** [https://huggingface.co/datasets/OpenMol/ChemCoTBench-CoT](https://huggingface.co/datasets/OpenMol/ChemCoTBench-CoT)

The evaluation script can refer to [baselines/evaluation_example.ipynb](baselines/evaluation_example.ipynb)
> NOTE: for jnk evaluation, the oracle pkl is old. So please change to some old env.

---
## 🧠 Evaluation Functions

To facilitate adoption, we have created an abstraction for the evaluation functions of the numerous subtasks spanning the four main tasks in ChemCoTBench. This offers researchers a streamlined process for rapid benchmarking through a single function call.

```
1. Molecule Understanding: baseline_and_eval/molund_eval_demo.ipynb
2. Molecule Editing: baseline_and_eval/moledit_eval_demo.ipynb
3. Molecule Optimization: baseline_and_eval/molopt_eval_demo.ipynb
4. Reactions: baseline_and_eval/rxn_eval_demo.ipynb
```

---
## 🤝 Contributing
We welcome contributions to enhance ChemCoTBench, including:
  - New chemical sources
  - 🧪 Additional chemical domains
  - 🧠 Novel evaluation tasks
  - 📝 Annotation improvements

---

## 📜 Citation
```bibtex
 @article{li2025beyond,
  title={Beyond Chemical QA: Evaluating LLM's Chemical Reasoning with Modular Chemical Operations},
  author={Li, Hao and Cao, He and Feng, Bin and Shao, Yanjun and Tang, Xiangru and Yan, Zhiyuan and Yuan, Li and Tian, Yonghong and Li, Yu},
  journal={arXiv preprint arXiv:2505.21318},
  year={2025}
}
```
