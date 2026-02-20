import os
import torch
import nltk
import pandas as pd
from nltk.corpus import stopwords
from docx import Document
from moverscore_v2 import get_idf_dict, word_mover_score

nltk.download('stopwords')

# --- DOCUMENTOS ---
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

# --- MÉTRICA ---
# Semantic similarity (MoverScore)
def mover_score(original, simplified, idf_dict_ref, idf_dict_hyp, use_stopwords=True):
  """
  Calcula MoverScore entre textos originales y adaptados
  Args:
      originals (list): Lista de textos originales (referencias)
      simplified (list): Lista de textos simplificados (hipótesis)
  Returns:
      list: Puntuaciones MoverScore para cada par de textos
  """

  try:
      os.environ['MOVERSCORE_MODEL'] = "dccuchile/bert-base-spanish-wwm-cased"

      if isinstance(original, str):
        original = [original]
      if isinstance(simplified, str):
        simplified = [simplified]

      # Validar textos no vacíos
      if not original[0].strip() or not simplified[0].strip():
        print("WARNING: Textos vacíos en MoverScore")
        return None

      spanish_stopwords = []
      if use_stopwords:
          try:
              spanish_stopwords = stopwords.words('spanish')
              # Stop words médicas comunes
              medical_stopwords = ['paciente', 'médico', 'doctor', 'hospital', 'clínica']
              spanish_stopwords.extend(medical_stopwords)
          except Exception as e:
              print(f"No se pudieron cargar stop words: {e}")
              spanish_stopwords = []

      #Calculo de puntuaciones (simplified = translations, originals = references)
      scores = word_mover_score(
          original,  # referencias
          simplified,  # hipótesis
          idf_dict_ref,
          idf_dict_hyp,
          stop_words=spanish_stopwords,
          n_gram=1,
          remove_subwords=True
      )
      return scores

  except ImportError:
      print("MoverScore no está disponible. Clona: git clone https://github.com/AIPHES/emnlp19-moverscore")
      return None
  except Exception as e:
      print(f"Error calculando MoverScore: {e}")
      return None


# === EVALUACIÓN DE MÉTRICAS YA IMPLEMENTADAS ===
def evaluate(name, original, adapted,  idf_dict_ref, idf_dict_hyp):
    resultado = {"documento": name}
    errores = []

    try:
        # MoverScore
        mover_result = mover_score(original, adapted, idf_dict_ref, idf_dict_hyp, use_stopwords=True)
        if mover_result is not None and isinstance(mover_result, list) and len(mover_result) > 0:
            resultado["MoverScore"] = round(mover_result[0], 4)
        else:
            resultado["MoverScore"] = None

    except Exception as e:
        print(f" Error en '{name}': {e}")
        errores.append(name)

    return resultado

# ---- MAIN ----
if __name__ == "__main__":
  doc_folder = "/opt/metricas/moverscore/Documentos"

  doc_pairs = match_documents(doc_folder)
  print(f"Pares encontrados: {len(doc_pairs)}")

  #Excel
  results_casos_clinicos = []
  results_cm_aapp = []
  errores = []

  #Lectura de textos para moverscore (IDF)
  all_originals = []
  all_adaptados = []
  pares_ordenados = []

  for name, paths in doc_pairs.items():
     try:
         texto_orig = read_docx_text(paths['original'])
         texto_lc = read_docx_text(paths['adapted'])

         #Guardamos para calcular IDF global
         all_originals.append(texto_orig)
         all_adaptados.append(texto_lc)
         #pares_ordenados.append((name, texto_orig, texto_lc))
         pares_ordenados.append((name, texto_orig, texto_lc, paths['adapted']))
     except Exception as e:
         print(f"Error leyendo archivos de {name}: {e}")
         errores.append(name)

  #Calcular IDF global moverscore
  idf_dict_ref = get_idf_dict(all_originals)
  idf_dict_hyp = get_idf_dict(all_adaptados)

  #Evaluar con IDF global ===
  for name, texto_orig, texto_lc, path_adapted in pares_ordenados:
    print(f"Evaluando: {name}")
    try:
        result = evaluate(name, texto_orig, texto_lc, idf_dict_ref, idf_dict_hyp)
        if "_LC" in path_adapted:
            results_casos_clinicos.append(result)
        elif "- LC" in path_adapted:
            results_cm_aapp.append(result)
    except Exception as e:
        print(f"Error en evaluación de {name}: {e}")
        errores.append(name)

  output_path = "/opt/metricas/moverscore/Metrics.xlsx"
  with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
      if results_casos_clinicos:
          pd.DataFrame(results_casos_clinicos).to_excel(writer, sheet_name="Casos clinicos", index=False)
      if results_cm_aapp:
          pd.DataFrame(results_cm_aapp).to_excel(writer, sheet_name="CM-AAPP", index=False)

  print("✅ Excel generado con hojas separadas.")