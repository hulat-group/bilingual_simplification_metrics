import os
import re
import spacy
import torch
import nltk
import pandas as pd
from nltk.corpus import stopwords
from docx import Document
# Ruta de import actualizada
from src.alignscore import AlignScore

# --- DOCUMENTOS ---
"""
---- COMO DEBE DE SER ----
original = texto original
reference = generado por un humano experto
adaptado = generado por un modelo de lenguaje

---- COMO SE TIENE ----
original = texto original
reference = generado por un modelo de lenguaje (_RE)
adaptado = generado por un humano experto (_LC)
"""


def sort_key_casos(texto):
    match = re.search(r'CasoClinico2020-(\d+)-(\d+)', texto)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (9999, 9999)  # fallback si no hay match

def read_docx_text(filepath):
    doc = Document(filepath)
    return "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])

def match_documents(doc_folder):
    files = [f for f in os.listdir(doc_folder) if f.endswith(".docx")]
    paired = {}

    for f in files:
      filename = f.replace(".docx", "")
      full_path = os.path.join(doc_folder, f)

      #case 1: Casos clinicos
      if filename.endswith("_LC"):
        base = filename.replace("_LC", "")
        if base not in paired:
          paired[base] = {}
        paired[base]["adapted"] = full_path

      #case 2: CM-AAPP
      elif filename.endswith("- LC"):
        base = filename.replace(" - LC", "")
        if base not in paired:
          paired[base] = {}
        paired[base]['adapted'] = full_path

      else:
        base = filename
        if base not in paired:
          paired[base] = {}
        paired[base]['original'] = full_path

    # Documentos Pares
    final_pairs = {k: v for k, v in paired.items() if 'original' in v and 'adapted' in v}
    print(f" Archivos detectados: {len(files)}")
    print(f"✅ Pares válidos encontrados: {len(final_pairs)}")
    return final_pairs

scorer = AlignScore(
    model="dccuchile/bert-base-spanish-wwm-cased",
    batch_size=32,
    device='cuda:2',
    ckpt_path="checkpoints_alignscore_es/dccuchile-bert-base-spanish-wwm-cased_es_final.ckpt",    
    #ckpt_path="checkpoints1_alignscore_es/dccuchile-bert-base-spanish-wwm-cased_final.ckpt",  
    evaluation_mode='nli_sp' 
  )

#Factual fidelity (AlignScore)
# Para ejecutar la librería, actualice los requistos parte del código del archivo model.py del repositorio original para que AdamW se importara correctamente desde la libreria transformers
def Align_Score(original, simplified):
  print("Calculando AlignScore...")
  #Modificar el path del checkpoint si es necesario, y demás parámetros si necesario, ahora mismo tiene los parámetros base
  #Hay que instalar los checkpoints usando unos comandos antes de ejecutar el programa
  #scorer = AlignScore(model='roberta-base', batch_size=32, device='cuda:0', ckpt_path='checkpoints/AlignScore-base.ckpt', evaluation_mode='nli_sp')
  score = scorer.score(contexts=[original], claims=[simplified])
  #score = scorer.score(contexts=['hello world.'], claims=['hello world.'])
  return score

# === EVALUACIÓN DE MÉTRICAS YA IMPLEMENTADAS ===
def evaluate(name, path_original, path_adapted):
    original = read_docx_text(path_original)
    adapted = read_docx_text(path_adapted)
    resultado = {"documento": name}
    errores = []

    try:
        #AlignScore
        align_score = Align_Score(original, adapted)
        resultado["AlignScore"] = round(align_score[0], 4)
        print(f" AlignScore: {resultado['AlignScore']}")

    except Exception as e:
        print(f" Error en '{name}': {e}")
        errores.append(name)

    return resultado

# ---- MAIN ----
if __name__ == "__main__":
  doc_folder = "./Documentos"

  doc_pairs = match_documents(doc_folder)
  print(f"Pares encontrados: {len(doc_pairs)}")

  results_casos_clinicos = []
  results_cm_aapp = []
  errores = []

  for name, paths in doc_pairs.items():
    print(f"Evaluando: {name}")
    result = evaluate(name, paths['original'], paths['adapted'])

    if name.endswith("_LC") or "_LC" in paths['adapted']:
      results_casos_clinicos.append(result)
    elif name.endswith("- LC") or "- LC" in paths['adapted']:
      results_cm_aapp.append(result)

  output_path = "./Metrics_alingscore_final1.xlsx"
  with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
      if results_casos_clinicos:
          #Ordenar por número de caso
          results_casos_clinicos.sort(key=lambda x: sort_key_casos(x["documento"]))
          pd.DataFrame(results_casos_clinicos).to_excel(writer, sheet_name="Casos clinicos", index=False)
      if results_cm_aapp:
          # Ordenar alfabéticamente
          results_cm_aapp.sort(key=lambda x: x["documento"].lower())
          pd.DataFrame(results_cm_aapp).to_excel(writer, sheet_name="CM-AAPP", index=False)

  print("✅ Excel generado con hojas separadas.")