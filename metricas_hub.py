# ****** Librerías estándar ******
import os
import sys
import re
import csv
import json
import nltk
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from docx import Document
from pathlib import Path
from huggingface_hub import login
from collections import defaultdict

# ****** Librerías externas ******
import torch
import pyphen
import evaluate
import textstat
import bleu
import summac.model_summac as summac_mod
from bert_score import score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from rouge import Rouge
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# **** Métricas especializadas ****
from AlignScore_v2_es.src.alignscore import AlignScore
from moverscore.moverscore_v2 import get_idf_dict, word_mover_score
from questeval.questeval_metric import QuestEval
from summac.model_summac import SummaCZS, SummaCConv
from codecarbon import OfflineEmissionsTracker

os.environ["TOKENIZERS_PARALLELISM"] = "false"
nltk.download('stopwords')

BASE_DIR = Path(__file__).resolve().parent

# ***************************
# ****** CONFIGURACIÓN ******
# ***************************
LANG_CFG = {
    "es": {
        "bertscore_model": "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
        "bertscore_lang": "es",
        "moverscore_model": "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
        "alignscore_model": "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
        "alignscore_eval_mode": "nli_sp",
        "alignscore_ckpt_path": None, # No se usa en este caso
        "summac_key": "mdeberta-es",
        "summac_model_card": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "readability": "fh",
    },
    "en": {
        "bertscore_model": "roberta-large",
        "bertscore_lang": "en",
        "moverscore_model": "roberta-large",
        "alignscore_model": "roberta-base",
        "alignscore_eval_mode": "nli_sp",
        "alignscore_ckpt_path": str(BASE_DIR / "AlignScore_v2_es" / "checkpoint" / "checkpoints" / "AlignScore-base.ckpt"),
        "summac_key": "tals/albert-xlarge-vitaminc-mnli",
        "summac_model_card": "tals/albert-xlarge-vitaminc-mnli",
        "readability": "flesch",
    }
}

# ************************
# ****** DOCUMENTOS ******
# ************************
"""
---- COMO ESTA IMPLEMENTADO ----
original = texto original (orginal)
reference = generado por un humano experto (references)
simplificado = generado por un modelo de lenguaje (simplified)
"""
def check_empty_files(paths):
    for p in paths:
        txt = read_file_text(p)
        if txt.strip() == "":
            print(f"❌ Archivo vacío o con solo espacios: {p}")

def read_docx_text(filepath):
    doc = Document(filepath)
    return "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])

def read_txt_text(filepath):
    """
    Lee archivos de texto plano (.txt)
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read().strip()

def read_file_text(filepath):
    """
    Lee archivos de texto según su extensión (.txt o .docx)
    """
    if filepath.endswith('.txt'):
        return read_txt_text(filepath)
    elif filepath.endswith('.docx'):
        return read_docx_text(filepath)
    else:
        raise ValueError(f"Tipo de archivo no soportado: {filepath}")

def match_documents(doc_folder_original, doc_folder_simplified, doc_folder_references1, doc_folder_references2, doc_folder_references3, n_refs=None):
    """
    Encuentra documentos base y sus tres referencias en carpetas separadas.
    
    Args:
        doc_folder_original: Carpeta con documentos originales/base
        doc_folder_references1: Carpeta con primera referencia
        doc_folder_references2: Carpeta con segunda referencia  
        doc_folder_references3: Carpeta con tercera referencia
    """
    paired = {}
    
    # Procesa los documentos originales
    if os.path.exists(doc_folder_original):
        files_original = [f for f in os.listdir(doc_folder_original) if f.endswith(".txt")]
        for f in files_original:
            filename = f.replace(".txt", "")
            full_path = os.path.join(doc_folder_original, f)
            
            if filename not in paired:
                paired[filename] = {}
            paired[filename]['original'] = full_path
    
    # Procesa los documentos Simplificados
    if os.path.exists(doc_folder_simplified):
        files_simplified = [f for f in os.listdir(doc_folder_simplified) if f.endswith(".txt")]
        for f in files_simplified:
            filename = f.replace(".txt", "")
            full_path = os.path.join(doc_folder_simplified, f)

            base = filename

            if base not in paired:
                paired[base] = {}
            paired[base]['simplified'] = full_path

    # Procesa la primera adaptación
    if os.path.exists(doc_folder_references1):
        files_references1 = [f for f in os.listdir(doc_folder_references1) if f.endswith(".txt")]
        for f in files_references1:
            filename = f.replace(".txt", "")
            full_path = os.path.join(doc_folder_references1, f)
            
            # Se limpian sufijos comunes
            if filename.endswith("_FA"):
                base = filename.replace("_FA", "")
            elif filename.endswith("-FA"):
                base = filename.replace("-FA", "")

            elif filename.endswith("_OGM"):
                base = filename.replace("_OGM", "")
            elif filename.endswith("-OGM"):
                base = filename.replace("-OGM", "")

            elif filename.endswith("_PIM"):
                base = filename.replace("_PIM", "")
            elif filename.endswith("-PIM"):
                base = filename.replace("-PIM", "")
            else:
                base = filename
            if base not in paired:
                paired[base] = {}
            if 'references' not in paired[base]:
                paired[base]['references'] = []
            paired[base]['references'].append(full_path)
  
    # Procesa la segunda adaptación
    if os.path.exists(doc_folder_references2):
        files_references2 = [f for f in os.listdir(doc_folder_references2) if f.endswith(".txt")]
        for f in files_references2:
            filename = f.replace(".txt", "")
            full_path = os.path.join(doc_folder_references2, f)
            
            # Se limpian sufijos comunes
            if filename.endswith("_FA"):
                base = filename.replace("_FA", "")
            elif filename.endswith("-FA"):
                base = filename.replace("-FA", "")

            elif filename.endswith("_OGM"):
                base = filename.replace("_OGM", "")
            elif filename.endswith("-OGM"):
                base = filename.replace("-OGM", "")

            elif filename.endswith("_PIM"):
                base = filename.replace("_PIM", "")
            elif filename.endswith("-PIM"):
                base = filename.replace("-PIM", "")
            else:
                base = filename               
            if base not in paired:
                paired[base] = {}
            if 'references' not in paired[base]:
                paired[base]['references'] = []
            paired[base]['references'].append(full_path)
    
    # Procesa la tercera adaptación
    if os.path.exists(doc_folder_references3):
        files_references3 = [f for f in os.listdir(doc_folder_references3) if f.endswith(".txt")]
        for f in files_references3:
            filename = f.replace(".txt", "")
            full_path = os.path.join(doc_folder_references3, f)
            
            # Se limpian sufijos comunes
            if filename.endswith("_FA"):
                base = filename.replace("_FA", "")
            elif filename.endswith("-FA"):
                base = filename.replace("-FA", "")

            elif filename.endswith("_OGM"):
                base = filename.replace("_OGM", "")
            elif filename.endswith("-OGM"):
                base = filename.replace("-OGM", "")

            elif filename.endswith("_PIM"):
                base = filename.replace("_PIM", "")
            elif filename.endswith("-PIM"):
                base = filename.replace("-PIM", "")
            else:
                base = filename            
            if base not in paired:
                paired[base] = {}
            if 'references' not in paired[base]:
                paired[base]['references'] = []
            paired[base]['references'].append(full_path)
    
    if n_refs is None:
        complete_sets = {k: v for k, v in paired.items()
                         if 'original' in v and 'simplified' in v and 'references' in v and len(v['references']) >= 1}
    else:
        complete_sets = {k: v for k, v in paired.items()
                         if 'original' in v and 'simplified' in v and 'references' in v and len(v['references']) == n_refs}
    
    # Se ordena alfabéticamente por nombre de archivo (CasoClinico36 → CasoClinico50)
    complete_sets = dict(sorted(complete_sets.items(), key=lambda x: x[0]))

    total_files = 0
    if os.path.exists(doc_folder_original):
        total_files += len([f for f in os.listdir(doc_folder_original) if f.endswith(".txt")])
    if os.path.exists(doc_folder_references1):
        total_files += len([f for f in os.listdir(doc_folder_references1) if f.endswith(".txt")])
    if os.path.exists(doc_folder_references2):
        total_files += len([f for f in os.listdir(doc_folder_references2) if f.endswith(".txt")])
    if os.path.exists(doc_folder_references3):
        total_files += len([f for f in os.listdir(doc_folder_references3) if f.endswith(".txt")])
    
    print(f" Archivos detectados: {total_files}")
    print(f"✅ Conjuntos completos encontrados: {len(complete_sets)}")
    return complete_sets

# ****** INICIALIZACIÓN DE SUMMAC ******
def init_summac(cfg):
    spanish_nli_config = {
        "model_card": cfg["summac_model_card"],
        "entailment_idx": 0,
        "contradiction_idx": 2,
    }
    summac_mod.model_map[cfg["summac_key"]] = spanish_nli_config

    print("Modelo registrado en SummaC:")
    print(summac_mod.model_map[cfg["summac_key"]])

    model_zs = SummaCZS(granularity="sentence", model_name=cfg["summac_key"], device="cuda")
    model_conv = SummaCConv(models=[cfg["summac_key"]], bins="percentile", granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")
    
    return model_zs, model_conv

# ****** INICIALIZACIÓN DE ALIGNSCORE ******
def init_alignscore(cfg, device="cuda:0", batch_size=32):
    """
    Inicializa AlignScore una sola vez desde cfg.
    """
    ckpt = cfg.get("alignscore_ckpt_path", None)

    scorer = AlignScore(model=cfg["alignscore_model"], batch_size=batch_size, ckpt_path=ckpt, device=device, evaluation_mode=cfg["alignscore_eval_mode"])

    return scorer

# ****** FUNCIONES DE MÉTRICAS ******

# -----------------------
# ------ BERTScore ------
# -----------------------
def calculate_bertscore(simplified, references, cfg):
    """
    Calcula el BERTScore
    """
    P, R, F1 = score([simplified], [references], model_type=cfg["bertscore_model"], num_layers=12, lang=cfg["bertscore_lang"], rescale_with_baseline=False)
    return P.mean().item(), R.mean().item(), F1.mean().item()

# ------------------------------
# ------ Readability ------
# ------------------------------
def calculate_readability(text, cfg):
    if cfg["readability"] == "fh":
        return {"fernandez_huerta": round(textstat.fernandez_huerta(text), 4)}
    elif cfg["readability"] == "flesch":
        return {
            "flesch_reading_ease": round(textstat.flesch_reading_ease(text), 4),
            #"fkgl": round(textstat.flesch_kincaid_grade(text), 4)
        }
    else:
        return {}

# ------------------
# ------ SARI ------
# ------------------
def calculate_sari(source, prediction, references):
    """
    Calcula el SARI (ADD, KEEP, DELETE) siguiendo el paper Optimizing Statistical Machine Translation for Text Simplification de Xu et al. (2016) y
    devuelve también la media de SARI.
    """
    def ngram_counter(sentence, n):
        """
         n-gramas en una oración.
        """
        return [tuple(sentence[i:i+n]) for i in range(len(sentence)-n+1)] #set

    def precision_recall_f1(tp, fp, fn):
        """
        Calcula precisión, recall y F1-score.
        """
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return precision, recall, f1

    # Separa en palabras
    source_tokens = source.split()
    prediction_tokens = prediction.split()
    references_tokens = [ref.split() for ref in references]

    # Defini n-gramas para diferentes niveles (1-gram, 2-gram, etc.)
    ngram_levels = [1, 2, 3, 4]

    add_scores, keep_scores, delete_scores = [], [], []

    for n in ngram_levels:
        source_ngrams = set(ngram_counter(source_tokens, n))
        pred_ngrams = set(ngram_counter(prediction_tokens, n))
        ref_ngrams = [set(ngram_counter(ref, n)) for ref in references_tokens]

        # Une todas las referencias
        union_ref_ngrams = set().union(*ref_ngrams)

        keep_ngrams = source_ngrams.intersection(pred_ngrams)
        add_ngrams = pred_ngrams.difference(source_ngrams)
        del_ngrams = source_ngrams.difference(pred_ngrams)

        # --- KEEP ---
        tp_keep = len(keep_ngrams.intersection(union_ref_ngrams))
        fp_keep = len(keep_ngrams.difference(union_ref_ngrams))
        fn_keep = len(source_ngrams.intersection(union_ref_ngrams).difference(keep_ngrams))

        # --- ADD ---
        tp_add = len(add_ngrams.intersection(union_ref_ngrams))
        fp_add = len(add_ngrams.difference(union_ref_ngrams))
        fn_add = len(union_ref_ngrams.difference(source_ngrams).difference(add_ngrams))

        # --- DELETE ---
        good_deletions = source_ngrams.difference(union_ref_ngrams)
        tp_del = len(del_ngrams.intersection(good_deletions))
        fp_del = len(del_ngrams.difference(good_deletions))

        keep_precision, keep_recall, keep_f1 = precision_recall_f1(tp_keep, fp_keep, fn_keep)
        add_precision, add_recall, add_f1 = precision_recall_f1(tp_add, fp_add, fn_add)
        del_precision = tp_del / (tp_del + fp_del) if (tp_del + fp_del) > 0 else 0
        del_f1 = del_precision

        add_scores.append(add_f1)
        keep_scores.append(keep_f1)
        delete_scores.append(del_f1)

    # Promedios finales
    avg_add = sum(add_scores) / len(add_scores)
    avg_keep = sum(keep_scores) / len(keep_scores)
    avg_del = sum(delete_scores) / len(delete_scores)
    sari = (avg_add + avg_keep + avg_del) / 3

    return avg_add, avg_keep, avg_del, sari

# --------------------------------------------------------------------
# ------ Semantic_Answer_Similarity (SAS sentence-transformers) ------
# --------------------------------------------------------------------
def SAS(simplified, references, st_model): #ENCODE PARA CADA REFERENCIA 
    """
    Devuelve un único SAS (cosine similarity) entre el texto simplificado y una referencia.
    """
    emb_simplified = st_model.encode([simplified], convert_to_tensor=True)
    emb_references = st_model.encode([references], convert_to_tensor=True)
    return util.cos_sim(emb_simplified, emb_references).item() 

# ------------------------
# ------ AlignScore ------
# ------------------------
def Align_Score(original, simplified, align_scorer):
    """
    Calcula AlignScore para (original, simplified) usando un scorer ya inicializado.
    """
    s = align_scorer.score(contexts=[original], claims=[simplified])

    # AlignScore suele devolver lista/array de longitud 1
    if isinstance(s, (list, tuple, np.ndarray)):
        return float(s[0])
    return float(s)

# ------------------------------------------
# ------ Factual fidelity (QuestEval) ------
# ------------------------------------------
def Quest_Eval(original, simplified, references):
    questeval = QuestEval(no_cuda=False)
    
    score = questeval.corpus_questeval(
        hypothesis=[simplified],
        sources=[original],
        list_references=references,
    )
    return score

# ------------------------
# ------ MoverScore ------
# ------------------------
def calculate_moverscore(simplified, references, idf_dict_ref, idf_dict_hyp, cfg, use_stopwords=False):
    """
    Calcula MoverScore entre el texto simplificado y CADA referencia.
    Devuelve una lista de scores (uno por referencia).
    """
    try:
        os.environ['MOVERSCORE_MODEL'] = cfg["moverscore_model"]

        if isinstance(references, str):
            references = [references]

        scores = word_mover_score(
            references,                      # lista de referencias
            [simplified] * len(references),  # hipótesis repetida
            idf_dict_ref,
            idf_dict_hyp,
            n_gram=1,
            remove_subwords=False
        )

        # word_mover_score devuelve una lista de floats
        return [float(s) for s in scores]

    except Exception as e:
        print(f"⚠️ Error en MoverScore: {e}")
        return [0.0] * len(references)

# ------------------
# ------ BLEU ------
# ------------------
def calculate_bleu(simplified, references):
    """
    Calcula métricas BLEU por referencia.
    Devuelve un diccionario donde cada clave es una lista de 3 valores (uno por referencia).
    """
    bleu_metric = evaluate.load("bleu")

    # Listas vacías
    bleu_scores = {
        "bleu": [],
        "bleu_precision-1": [],
        "bleu_precision-2": [],
        "bleu_precision-3": [],
        "bleu_precision-4": [],
        "bleu_brevity_penalty": [],
        "bleu_length_ratio": [],
        "bleu_translation_length": [],
        "bleu_reference_length": [],
    }

    for ref in references:
        # references debe ser lista de listas: [[ref]]
        results = bleu_metric.compute(predictions=[simplified], references=[[ref]])

        bleu_scores["bleu"].append(results["bleu"])
        bleu_scores["bleu_precision-1"].append(results["precisions"][0])
        bleu_scores["bleu_precision-2"].append(results["precisions"][1])
        bleu_scores["bleu_precision-3"].append(results["precisions"][2])
        bleu_scores["bleu_precision-4"].append(results["precisions"][3])
        bleu_scores["bleu_brevity_penalty"].append(results["brevity_penalty"])
        bleu_scores["bleu_length_ratio"].append(results["length_ratio"])
        bleu_scores["bleu_translation_length"].append(results["translation_length"])
        bleu_scores["bleu_reference_length"].append(results["reference_length"])
    return bleu_scores

# -------------------
# ------ ROUGE ------
# -------------------
def calculate_rouge(simplified, references):
    """
    Calcula métricas ROUGE detalladas
    """
    rouge = Rouge()

    rouge_scores = {
        "rouge-1": {
            "rouge-1_f1": [],
            "rouge-1_precision": [],
            "rouge-1_recall": []
        },
        "rouge-2": {
            "rouge-2_f1": [],
            "rouge-2_precision": [],
            "rouge-2_recall": []
        },
        "rouge-l": {
            "rouge-l_f1": [],
            "rouge-l_precision": [],
            "rouge-l_recall": []
        }
    }

    for ref in references:
        scores = rouge.get_scores(simplified, ref, avg=True)
        rouge_scores["rouge-1"]["rouge-1_f1"].append(scores['rouge-1']['f'])
        rouge_scores["rouge-1"]["rouge-1_precision"].append(scores['rouge-1']['p'])
        rouge_scores["rouge-1"]["rouge-1_recall"].append(scores['rouge-1']['r'])

        rouge_scores["rouge-2"]["rouge-2_f1"].append(scores['rouge-2']['f'])
        rouge_scores["rouge-2"]["rouge-2_precision"].append(scores['rouge-2']['p'])
        rouge_scores["rouge-2"]["rouge-2_recall"].append(scores['rouge-2']['r'])

        rouge_scores["rouge-l"]["rouge-l_f1"].append(scores['rouge-l']['f'])
        rouge_scores["rouge-l"]["rouge-l_precision"].append(scores['rouge-l']['p'])
        rouge_scores["rouge-l"]["rouge-l_recall"].append(scores['rouge-l']['r'])

    return rouge_scores

# --------------------
# ------ SummaC ------
# --------------------
def calculate_summac(summac_zs_model, summac_conv_model, original, simplified):
    """
    Implementación de SummaC 
    """   
    score_zs = summac_zs_model.score([original], [simplified])
    score_conv = summac_conv_model.score([original], [simplified])
    return score_zs["scores"][0], score_conv["scores"][0]

# ***************************************
# ****** EVALUACIÓN REESTRUCTURADA ******
# ***************************************
def evaluate_pair(name, path_original, path_simplified, path_references, st_model, idf_dict_ref, idf_dict_hyp, summac_zs_model, summac_conv_model, align_scorer, cfg):

    original = read_file_text(path_original)
    simplified = read_file_text(path_simplified)

    check_empty_files(path_references)
    references_text = [read_file_text(p) for p in path_references]

    # ====== REFERENCIAS ======
    reference_ids = [Path(p).stem for p in path_references]

    # ====== MÉTRICAS POR REFERENCIA ======
    # --- SARI ---
    sari_add = []
    sari_keep = []
    sari_del = []
    sari_total = []

    for ref in references_text:
        add, keep, delete, sari = calculate_sari(original, simplified, [ref])
        sari_add.append(round(add, 4))
        sari_keep.append(round(keep, 4))
        sari_del.append(round(delete, 4))
        sari_total.append(round(sari, 4))

    # --- BERTScore (3 valores) ---
    bert_P = []
    bert_R = []
    bert_F1 = []
    for ref in references_text:
        P, R, F1 = calculate_bertscore(simplified, ref, cfg)
        bert_P.append(round(P, 4))
        bert_R.append(round(R, 4))
        bert_F1.append(round(F1, 4))

    # --- SAS (3 valores) ---
    sas_values = [round(SAS(simplified, ref, st_model), 4)
                  for ref in references_text]

    # --- MoverScore (3 valores) ---
    moverscores = calculate_moverscore(simplified, references_text,
                                       idf_dict_ref, idf_dict_hyp, cfg)

    # --- BLEU (3 valores por métrica) ---
    bleu_vals = {"bleu": [], "bleu_precision-1": [], "bleu_precision-2": [],
                "bleu_precision-3": [], "bleu_precision-4": [],
                "bleu_brevity_penalty": [], "bleu_length_ratio": [],
                "bleu_translation_length": [], "bleu_reference_length": []}

    for ref in references_text:
        b = calculate_bleu(simplified, [ref])  
        for k in bleu_vals:
            bleu_vals[k].append(round(b[k][0], 4)) 

    # --- ROUGE (3 valores por métrica) ---
    rouge_vals = {
        "rouge-1_f1": [], "rouge-1_precision": [], "rouge-1_recall": [],
        "rouge-2_f1": [], "rouge-2_precision": [], "rouge-2_recall": [],
        "rouge-l_f1": [], "rouge-l_precision": [], "rouge-l_recall": []
    }

    for ref_path, ref in zip(path_references, references_text):

        # En caso de un archivo este vacío se mostrará un mensaje
        if ref.strip() == "":
            print("\n❌❌ REFERENCIA VACÍA DETECTADA")
            print("Archivo:", ref_path)
            print("Contenido leído:", repr(ref))
            print("----------------------------------------\n")

        try:
            r = calculate_rouge(simplified, [ref]) 

        except Exception as e:
            print("\n🔥 ERROR EN ROUGE")
            print("Archivo de referencia que produjo el error:", ref_path)
            print("Contenido:", repr(ref))
            print("Error:", e)
            print("------------------------------------------------\n")
            raise e  # Re-lanzar para detener ejecución

        # guardar resultados
        rouge_vals["rouge-1_f1"].append(round(r["rouge-1"]["rouge-1_f1"][0], 4))
        rouge_vals["rouge-1_precision"].append(round(r["rouge-1"]["rouge-1_precision"][0], 4))
        rouge_vals["rouge-1_recall"].append(round(r["rouge-1"]["rouge-1_recall"][0], 4))

        rouge_vals["rouge-2_f1"].append(round(r["rouge-2"]["rouge-2_f1"][0], 4))
        rouge_vals["rouge-2_precision"].append(round(r["rouge-2"]["rouge-2_precision"][0], 4))
        rouge_vals["rouge-2_recall"].append(round(r["rouge-2"]["rouge-2_recall"][0], 4))

        rouge_vals["rouge-l_f1"].append(round(r["rouge-l"]["rouge-l_f1"][0], 4))
        rouge_vals["rouge-l_precision"].append(round(r["rouge-l"]["rouge-l_precision"][0], 4))
        rouge_vals["rouge-l_recall"].append(round(r["rouge-l"]["rouge-l_recall"][0], 4))

    # --- ALIGNScore → valor único (media) ---
    alignscore = round(Align_Score(original, simplified, align_scorer), 4)

    # --- SummaC ---
    summac_zs, summac_conv = calculate_summac(summac_zs_model, summac_conv_model, original, simplified)

    # --- QuestEval (3 valores) ---
    quest_vals = []
    for ref in references_text:
        q = Quest_Eval(original, simplified, [ref])
        quest_vals.append(round(q["corpus_score"], 4))

    # --- Readability ---
    readability_original = calculate_readability(original, cfg)
    readability_refs = [calculate_readability(r, cfg) for r in references_text]
    readability_simplified = calculate_readability(simplified, cfg)

    # =============
    #      JSON 
    # =============
    
    result = {
        "id_original_text": name,
        "original_text": original,
        "id_reference_text": reference_ids,
        "reference_text": references_text,
        "simplified_text": simplified,
        "model_name": MODEL_NAME,
        "scores": {
            "simplification": {
                "sari": {
                    "sari_total": sari_total,
                    "sari_add": sari_add,
                    "sari_keep": sari_keep,
                    "sari_del": sari_del
                }
            },
            "similarity": {
                "bertscore": {
                    "bertscore_precision": bert_P,
                    "bertscore_recall": bert_R,
                    "bertscore_f1": bert_F1
                },
                "moverscore": moverscores,
                "sas": sas_values,
                "bleu": bleu_vals,
                "rouge": rouge_vals
            },
            "factuality": {
                "alignscore": alignscore,
                "summac": {
                    "summac_zs": round(summac_zs, 4),
                    "summac_conv": round(summac_conv, 4)
                },
                "questeval": quest_vals
            },
            "readability": {
                "original": readability_original,
                "references": readability_refs,
                "simplified": readability_simplified
            }
        }
    }

    return [result]

# **************************
# ****** EXPORTAR CSV ******
# **************************
def results_to_csv(results, output_path):

    csv_rows = []

    for r in results:
        refs = r["id_reference_text"] 
        sari = r["scores"]["simplification"]["sari"]
        bert = r["scores"]["similarity"]["bertscore"]
        bleu = r["scores"]["similarity"]["bleu"]
        rouge = r["scores"]["similarity"]["rouge"]
        mover = r["scores"]["similarity"]["moverscore"]
        sas_st = r["scores"]["similarity"]["sas"]
        quest = r["scores"]["factuality"]["questeval"]
        readability_refs = r["scores"]["readability"]["references"]

        for i in range(len(refs)):
            ro = r["scores"]["readability"]["original"]
            rr = r["scores"]["readability"]["references"][i]
            rs = r["scores"]["readability"]["simplified"]

            row = {
                "id_original_text": r["id_original_text"],
                "id_reference_text": refs[i],
                "model_name": r["model_name"],

                "sari_total": sari["sari_total"][i],
                "sari_add": sari["sari_add"][i],
                "sari_keep": sari["sari_keep"][i],
                "sari_del": sari["sari_del"][i],

                "bertscore_precision": bert["bertscore_precision"][i],
                "bertscore_recall": bert["bertscore_recall"][i],
                "bertscore_f1": bert["bertscore_f1"][i],

                "moverscore": mover[i],
                "sas": sas_st[i],

                "bleu": bleu["bleu"][i],
                "bleu_precision_1": bleu["bleu_precision-1"][i],
                "bleu_precision_2": bleu["bleu_precision-2"][i],
                "bleu_precision_3": bleu["bleu_precision-3"][i],
                "bleu_precision_4": bleu["bleu_precision-4"][i],
                "bleu_brevity_penalty": bleu["bleu_brevity_penalty"][i],
                "bleu_length_ratio": bleu["bleu_length_ratio"][i],
                "bleu_translation_length": bleu["bleu_translation_length"][i],
                "bleu_reference_length": bleu["bleu_reference_length"][i],

                "rouge_1_f1": rouge["rouge-1_f1"][i],
                "rouge_1_precision": rouge["rouge-1_precision"][i],
                "rouge_1_recall": rouge["rouge-1_recall"][i],

                "rouge_2_f1": rouge["rouge-2_f1"][i],
                "rouge_2_precision": rouge["rouge-2_precision"][i],
                "rouge_2_recall": rouge["rouge-2_recall"][i],

                "rouge_l_f1": rouge["rouge-l_f1"][i],
                "rouge_l_precision": rouge["rouge-l_precision"][i],
                "rouge_l_recall": rouge["rouge-l_recall"][i],

                "alignscore": r["scores"]["factuality"]["alignscore"],
                "summac_zs": r["scores"]["factuality"]["summac"]["summac_zs"],
                "summac_conv": r["scores"]["factuality"]["summac"]["summac_conv"],
                "questeval": quest[i],

                "readability_original": list(ro.values())[0] if ro else None,
                "readability_reference": list(rr.values())[0] if rr else None,
                "readability_simplified": list(rs.values())[0] if rs else None,
            }

            csv_rows.append(row)

    df = pd.DataFrame(csv_rows)
    df.to_csv(output_path, index=False, sep=";", encoding="utf-8", decimal=",")

# **************************
# *** PARSEAR ARGUMENTOS ***
# **************************
def parse_args():
    parser = argparse.ArgumentParser(description="Bilingual metrics (ES/EN)")
    parser.add_argument("--lang", choices=["es", "en"], required=True,
                        help="Idioma de evaluación: es | en")
    parser.add_argument("--data_root", required=True,
                        help="Raíz del dataset (contiene originals_txt_test/ y references_txt_test/)")
    parser.add_argument("--simpl_root", required=True,
                        help="Raíz de simplificaciones (contiene subcarpetas por modelo)")
    parser.add_argument("--out", required=True,
                        help="Carpeta de salida para JSON/CSV/CodeCarbon")
    return parser.parse_args()

# **********************
# ****** MAIN CSV ******
# **********************
if __name__ == "__main__":

    args = parse_args()
    cfg = LANG_CFG[args.lang]

    data_root = Path(args.data_root)
    DOC_FOLDER_ORIGINAL = str(data_root / "")

    # Referencias: en español/ingles, el nombre de reference_x, como generico
    if args.lang == "es":
        ref_dirs = [
            data_root / "references_txt_test" / "reference_1",
            data_root / "references_txt_test" / "reference_2",
            data_root / "references_txt_test" / "reference_3",
        ]
    else:
        ref_dirs = [
            data_root / "references_txt_test" / "reference_1",
            data_root / "references_txt_test" / "reference_1",
            data_root / "references_txt_test" / "reference_1",
        ]
    DOC_FOLDER_REFERENCES1 = str(ref_dirs[0])
    DOC_FOLDER_REFERENCES2 = str(ref_dirs[1])
    DOC_FOLDER_REFERENCES3 = str(ref_dirs[2])

    st_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    # SummaC
    model_zs_summac, model_conv_summac = init_summac(cfg)
    align_scorer = init_alignscore(cfg, device="cuda:0", batch_size=32)
   
    # -----------------------------
    # Normalizacion de raíz de simplificaciones:
    # --simpl_root puede ser .../Conjunto_Simplificaciones
    # -----------------------------

    simpl_root = Path(args.simpl_root)
    if (simpl_root / "es").is_dir() or (simpl_root / "en").is_dir():
        simpl_root = simpl_root / args.lang

    SIMPLIFICACIONES_PATH = str(simpl_root)

    OUTPUT_FOLDER = args.out
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    MODELS = sorted([d for d in os.listdir(SIMPLIFICACIONES_PATH) if os.path.isdir(os.path.join(SIMPLIFICACIONES_PATH, d))])

    print(f"📁 Directorios detectados en {SIMPLIFICACIONES_PATH}: {MODELS}\n")
    print(f"📂 Resultados se guardarán en: {OUTPUT_FOLDER}\n")

    emissions_path_global = os.path.join(OUTPUT_FOLDER, "emissions_metricas_GLOBAL.csv")
    tracker_global = OfflineEmissionsTracker(
        project_name="metricas_GLOBAL",
        output_file=emissions_path_global,
        country_iso_code="ESP",
    )

    # Temporizador global
    total_start = time.time()
    tracker_global.start()

    try:    
        # Procesar cada modelo 
        for MODEL_NAME in tqdm(MODELS, desc="Procesando modelos", unit="modelo", position=0, leave=True):
            start_time = time.time()

            print("\n" + "="*90)
            print(f"🚀 Procesando modelo: {MODEL_NAME}")
            print("="*90)

            emissions_path_model = os.path.join(OUTPUT_FOLDER, f"emissions_metricas_{MODEL_NAME}.csv")        
            output_path_json = os.path.join(OUTPUT_FOLDER, f"metrics_{MODEL_NAME}.json")
            output_path_csv = os.path.join(OUTPUT_FOLDER, f"metrics_{MODEL_NAME}.csv")
            
            # ==========================
            # TRACKER POR MODELO
            # ==========================
            
            tracker_model = OfflineEmissionsTracker(
                project_name=f"metricas_{MODEL_NAME}",
                output_file=emissions_path_model,
                country_iso_code="ESP",
            )

            tracker_model.start()
            all_results = []

            try:
                DOC_FOLDER_SIMPLIFIED = os.path.join(SIMPLIFICACIONES_PATH, MODEL_NAME)            

                # Se emparejan los documentos originales, simplificados y referencias
                n_refs = 3
                doc_pairs = match_documents(DOC_FOLDER_ORIGINAL, DOC_FOLDER_SIMPLIFIED, DOC_FOLDER_REFERENCES1, DOC_FOLDER_REFERENCES2, DOC_FOLDER_REFERENCES3, n_refs=n_refs)
               
                print(f"Pares encontrados: {len(doc_pairs)}")

                # IDF GLOBAL 
                all_originals, all_simplified = [], []
                for n, p in doc_pairs.items():
                    all_originals.append(read_file_text(p['original']))
                    all_simplified.append(read_file_text(p['simplified']))

                idf_dict_ref = get_idf_dict(all_originals)
                idf_dict_hyp = get_idf_dict(all_simplified)

                for name, paths in tqdm(sorted(doc_pairs.items(), key=lambda x: x[0]), desc=f"Evaluando {MODEL_NAME}", unit="caso", position=1, leave=False):
                    pair_results = evaluate_pair(name, paths['original'], paths['simplified'], 
                                            paths['references'], st_model, idf_dict_ref, idf_dict_hyp, model_zs_summac, model_conv_summac, align_scorer, cfg)
                    all_results.extend(pair_results)
                    
                all_results = sorted(all_results, key=lambda r: r['id_original_text'])

                # Guardar resultados en JSON
                with open(output_path_json, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)

                # Guardar resultados en CSV
                results_to_csv(all_results, output_path_csv)

            finally:
                tracker_model.stop()

                # Normalizar CSV de CodeCarbon (modelo) a formato europeo
                if os.path.exists(emissions_path_model):
                    df_em = pd.read_csv(emissions_path_model)
                    df_em.to_csv(emissions_path_model, sep=";", decimal=",", index=False, encoding="utf-8")
                    print(f"✅ Emisiones ({MODEL_NAME}) guardadas (formato europeo): {emissions_path_model}")

            # Tiempo por modelo 
            elapsed = time.time() - start_time
            mins, secs = divmod(elapsed, 60)
            print(f"✅ Resultados guardados en {output_path_json}")
            print(f"✅ Resultados guardados en {output_path_csv}")
            print(f"📊 Total de evaluaciones: {len(all_results)}")
            print(f"⏱️ Tiempo de ejecución para {MODEL_NAME}: {int(mins)} min {int(secs)} s\n")
    
    finally:
        tracker_global.stop()

        # Normalizar CSV de CodeCarbon (global) a formato europeo
        if os.path.exists(emissions_path_global):
            df_em = pd.read_csv(emissions_path_global)
            df_em.to_csv(emissions_path_global, sep=";", decimal=",", index=False, encoding="utf-8")
            print(f"✅ Emisiones GLOBAL guardadas (formato europeo): {emissions_path_global}")

        # Tiempo total
        total_elapsed = time.time() - total_start
        total_mins, total_secs = divmod(total_elapsed, 60)
        total_hours, total_mins = divmod(total_mins, 60)
        print("=" * 90)
        print(f"🏁 Proceso completo finalizado.")
        print(f"🕒 Tiempo total: {int(total_hours)} h {int(total_mins)} min {int(total_secs)} s")
        print("=" * 90)
