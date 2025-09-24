from abc import ABC, abstractmethod
from typing import List, Dict
from functools import partial

from Levenshtein import distance as lev
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, AllChem
import selfies as sf
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score

RDLogger.DisableLog('rdApp.*')
nltk.download('wordnet')


def exact_match(ot_smi, gt_smi):
    m_out = Chem.MolFromSmiles(ot_smi)
    m_gt = Chem.MolFromSmiles(gt_smi)

    try:
        if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt):
            return 1
    except:
        pass
    return 0


def maccs_similarity(ot_m, gt_m):
    return DataStructs.FingerprintSimilarity(
        MACCSkeys.GenMACCSKeys(gt_m), 
        MACCSkeys.GenMACCSKeys(ot_m), 
        metric=DataStructs.TanimotoSimilarity
    )

def morgan_similarity(ot_m, gt_m, radius=2):
    return DataStructs.TanimotoSimilarity(
        AllChem.GetMorganFingerprint(gt_m, radius), 
        AllChem.GetMorganFingerprint(ot_m, radius)
    )

def rdk_similarity(ot_m, gt_m):
    return DataStructs.FingerprintSimilarity(
        Chem.RDKFingerprint(gt_m), 
        Chem.RDKFingerprint(ot_m), 
        metric=DataStructs.TanimotoSimilarity
    )


class Evaluator(ABC):

    @abstractmethod
    def build_evaluate_tuple(self, pred, gt):
        pass

    @abstractmethod
    def evaluate(self, predictions, references, metrics: List[str] = None, verbose: bool = False, full_results: bool = False):
        pass


class ClassificationEvaluator(Evaluator):
    _metric_functions = {
        "accuracy": accuracy_score,
        "f1_macro": partial(f1_score, average='macro'),
    }

    def build_evaluate_tuple(self, pred, gt):
        return pred, gt

    def evaluate(self, predictions, references, metrics: List[str] = None, verbose: bool = False, full_results: bool = False):
        if metrics is None:
            metrics = ["accuracy", "f1_macro", "f1_micro"]

        results = {metric: [] for metric in metrics}

        pred, gt = self.build_evaluate_tuple(predictions, references)

        for metric in metrics:
            results[metric].append(self._metric_functions[metric](gt, pred))

        if verbose:
            print("Evaluation results:")
            for metric, values in results.items():
                print(f"{metric}: {np.mean(values)}")

        if full_results:
            return results
        else:
            return {metric: np.mean(values) for metric, values in results.items()}


class RegressionEvaluator(Evaluator):
    _metric_functions = {
        "mae": mean_absolute_error,
        "mse": mean_squared_error,
        "r2": r2_score
    }

    def build_evaluate_tuple(self, pred, gt):
        return pred, gt

    def evaluate(self, predictions, references, metrics: List[str] = None, verbose: bool = False, full_results: bool = False):
        if metrics is None:
            metrics = ["mae", "mse", "r2"]

        results = {metric: [] for metric in metrics}
        gt, pred = self.build_evaluate_tuple(predictions, references) 

        for metric in metrics:
            results[metric].append(self._metric_functions[metric](gt, pred))

        if verbose:
            print("Evaluation results:")
            for metric, values in results.items():
                print(f"{metric}: {np.mean(values)}")

        if full_results:
            return results
        else:
            return {metric: np.mean(values) for metric, values in results.items()}


class MoleculeSMILESEvaluator(Evaluator):
    _metric_functions = {
        "exact_match": exact_match,
        "bleu": corpus_bleu,
        "levenshtein": lev,
        "rdk_sims": rdk_similarity,
        "maccs_sims": maccs_similarity,
        "morgan_sims": morgan_similarity,
        "validity": lambda smiles: smiles is not None,
    }

    @staticmethod
    def sf_decode(selfies):
        try:
            smiles = sf.decoder(selfies)
            return smiles
        except Exception:
            return None

    @staticmethod
    def convert_to_canonical_smiles(smiles):
        if not smiles:
            return None
        try:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is not None:
                canonical_smiles = Chem.MolToSmiles(molecule, isomericSmiles=False, canonical=True)
                return canonical_smiles
            else:
                return None
        except Exception:
            return None

    def build_evaluate_tuple(self, pred, gt, decode_selfies=False):
        if decode_selfies:
            pred = self.sf_decode(pred)
            gt = self.sf_decode(gt)
        return self.convert_to_canonical_smiles(pred), self.convert_to_canonical_smiles(gt)

    def evaluate(self, predictions, references, metrics: List[str] = None, verbose: bool = False, decode_selfies: bool = False, full_results: bool = False):
            
        if metrics is None:
            metrics = ["exact_match", "bleu", "levenshtein", "rdk_sims", "maccs_sims", "morgan_sims", "validity"]

        results = {metric: [] for metric in metrics}
        if "bleu" in metrics:
            results["bleu"] = [[], []]

        for pred, gt in zip(predictions, references):
            pred, gt = self.build_evaluate_tuple(pred, gt, decode_selfies=decode_selfies)

            for metric in metrics:
                if metric == "bleu" and pred and gt:
                    gt_tokens = [c for c in gt]
                    pred_tokens = [c for c in pred]
                    results[metric][0].append([gt_tokens])
                    results[metric][1].append(pred_tokens)
                elif pred is None or gt is None:
                    results[metric].append(0)
                    continue
                elif metric == "validity":
                    results[metric].append(self._metric_functions[metric](pred))
                elif metric in ["maccs_sims", "morgan_sims", "rdk_sims"]:
                    results[metric].append(self._metric_functions[metric](Chem.MolFromSmiles(pred), Chem.MolFromSmiles(gt)))
                else:
                    results[metric].append(self._metric_functions[metric](pred, gt))

        if "bleu" in metrics:
            if results["bleu"][0] and results["bleu"][1]:
                results["bleu"] = corpus_bleu(results["bleu"][0], results["bleu"][1])
            else:
                results["bleu"] = 0

        if verbose:
            print("Evaluation results:")
            for metric, values in results.items():
                print(f"{metric}: {np.mean(values)}")

        if full_results:
            return results
        else:
            return {metric: np.mean(values) for metric, values in results.items()}


class MoleculeSELFIESEvaluator(Evaluator):
    """
    If input is SMILES, convert to SELFIES then evaluate. Ensure `encode_smiles`=True
    """
    _metric_functions = {
        "levenshtein": lev,
        "exact_match": exact_match,
        "bleu": corpus_bleu,
        "validity": lambda selfies: selfies is not None,
        "maccs_sims": maccs_similarity,
        "morgan_sims": morgan_similarity,
        "rdk_sims": rdk_similarity
    }

    @staticmethod
    def sf_decode(selfies):
        try:
            smiles = sf.decoder(selfies)
            return smiles
        except Exception:
            return None
    
    @staticmethod
    def sf_encode(smiles):
        try:
            selfies = sf.encoder(smiles)
            return selfies
        except Exception:
            return None

    @staticmethod
    def convert_to_canonical_smiles(smiles):
        if not smiles:
            return None
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            canonical_smiles = Chem.MolToSmiles(molecule, isomericSmiles=False, canonical=True)
            return canonical_smiles
        else:
            return None

    def build_evaluate_tuple(self, pred, gt, encode_smiles=False):
        if encode_smiles:
            pred = self.convert_to_canonical_smiles(pred)
            gt = self.convert_to_canonical_smiles(gt)
            pred_sf = self.sf_encode(pred)
            gt_sf = self.sf_encode(gt)
        else:
            pred_sf, gt_sf = pred, gt
        return pred_sf, gt_sf, pred, gt

    def evaluate(self, predictions, references, metrics: List[str] = None, verbose: bool = False, encode_smiles: bool = False, full_results: bool = False):
            
        if metrics is None:
            metrics = ["levenshtein", "exact_match", "bleu", "validity", "maccs_sims", "morgan_sims", "rdk_sims"]

        results = {metric: [] for metric in metrics}
        if "bleu" in metrics:
            results["bleu"] = [[], []]

        for pred, gt in zip(predictions, references):
            pred, gt, pred_smi, gt_smi = self.build_evaluate_tuple(pred, gt, encode_smiles=encode_smiles)

            for metric in metrics:
                if metric == "bleu" and pred and gt:
                    gt_tokens = [c for c in gt]
                    pred_tokens = [c for c in pred]
                    results[metric][0].append([gt_tokens])
                    results[metric][1].append(pred_tokens)
                elif pred is None or gt is None:
                    results[metric].append(0)
                    continue
                elif metric == "validity":
                    results[metric].append(self._metric_functions[metric](pred))
                elif metric in ["maccs_sims", "morgan_sims", "rdk_sims"]:
                    results[metric].append(self._metric_functions[metric](Chem.MolFromSmiles(pred_smi), Chem.MolFromSmiles(gt_smi)))
                else:
                    results[metric].append(self._metric_functions[metric](pred_smi, gt_smi))

        if "bleu" in metrics:
            if results["bleu"][0] and results["bleu"][1]:
                results["bleu"] = corpus_bleu(results["bleu"][0], results["bleu"][1])
            else:
                results["bleu"] = 0

        if verbose:
            print("Evaluation results:")
            for metric, values in results.items():
                print(f"{metric}: {np.mean(values)}")

        if full_results:
            return results
        else:
            return {metric: np.mean(values) for metric, values in results.items()}


class MoleculeCaptionEvaluator(Evaluator):
    _metric_functions = {
        "bleu-2": partial(sentence_bleu, weights=(0.5, 0.5)),
        "bleu-4": partial(sentence_bleu, weights=(0.25, 0.25, 0.25, 0.25)),
        "meteor": meteor_score,
    }

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def build_evaluate_tuple(self, pred, gt):
        return pred, gt

    def evaluate(self, predictions, references, metrics: List[str] = None, verbose: bool = False, full_results: bool = False):
        if metrics is None:
            metrics = ["bleu-2", "bleu-4", "meteor", "rouge-1", "rouge-2", "rouge-L"]

        results = {metric: [] for metric in metrics}

        for pred, gt in zip(predictions, references):
            pred, gt = self.build_evaluate_tuple(pred, gt)

            for metric in metrics:
                if metric in ["bleu-2", "bleu-4"]:
                    results[metric].append(self._metric_functions[metric]([gt], pred))
                elif metric == "meteor":
                    results[metric].append(self._metric_functions[metric]([word_tokenize(gt)], word_tokenize(pred)))
                elif metric.startswith("rouge"):
                    scores = self.rouge_scorer.score(gt, pred)
                    rouge_variant = metric.split("-")[-1]
                    score = scores['rouge' + rouge_variant].fmeasure
                    results[metric].append(score)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
                if metric in ["bleu-2", "bleu-4"]:
                    results[metric].append(self._metric_functions[metric]([gt], pred))
                elif metric == "meteor":
                    results[metric].append(self._metric_functions[metric]([word_tokenize(gt)], word_tokenize(pred)))
                elif metric.startswith("rouge"):
                    scores = self.rouge_scorer.score(gt, pred)
                    rouge_variant = metric.split("-")[-1]
                    score = scores['rouge' + rouge_variant].fmeasure
                    results[metric].append(score)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")

        if verbose:
            print("Evaluation results:")
            for metric, values in results.items():
                print(f"{metric}: {np.mean(values)}")

        if full_results:
            return results
        else:
            return {metric: np.mean(values) for metric, values in results.items()}