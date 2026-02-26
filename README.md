# 🌍 Bilingual Simplification Metrics  
### 🧠 A Unified Evaluation Framework for Text Simplification (ES/EN)

This repository integrates structural simplification metrics, semantic similarity, factual consistency evaluation, readability, and environmental impact tracking into a single unified pipeline.

---

## ✨ Key Features

🔹 **Bilingual support** (Spanish 🇪🇸 / English 🇬🇧)  
🔹 Structured evaluation across multiple models  
🔹 JSON + CSV export   
🔹 CodeCarbon integration 

---

# 📊 Implemented Metrics
The repository provides a unified evaluation pipeline. However, some metrics internally depend on **different models depending on the language**, configured through `LANG_CFG` inside `metricas_hub.py`.

---

## 🧪 Metric–Model Configuration by Language

| Category | Metric | Spanish (ES) | English (EN) |
|-----------|--------|--------------|--------------|
| 🪄 Simplification | **SARI** | Rule-based implementation | Rule-based implementation |
| 🔁 Semantic Similarity | **BERTScore** | `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es` | `roberta-large` |
| 🔁 Semantic Similarity | **MoverScore** | `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es` | `roberta-large` |
| 🔁 Semantic Similarity | **SAS** | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | Same multilingual model |
| 🔁 Semantic Similarity | **BLEU** | `evaluate`  | `evaluate`  |
| 🔁 Semantic Similarity | **ROUGE** | `rouge`  | `rouge`  |
| 🧠 Factual Consistency | **AlignScore** | `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es` | `roberta-base` + AlignScore checkpoint |
| 🧠 Factual Consistency | **SummaC** | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | `tals/albert-xlarge-vitaminc-mnli` |
| 🧠 Factual Consistency | **QuestEval** | English QA/QG models | English QA/QG models |
| 📚 Readability | Readability Index | Fernández-Huerta | Flesch Reading Ease |

---

## 🔎 Important Clarification

Although the framework is described as *bilingual*, this refers to:

> A **single unified evaluation pipeline** supporting multiple languages through language-specific model configurations.

All model selection logic is defined in:

```python
LANG_CFG = {
    "es": {...},
    "en": {...}
}
```

## 🌱 Environmental Impact
- **CodeCarbon** (offline emissions tracking)

---

# 🏗 Repository Structure

```text
bilingual_simplification_metrics/
│
├── metricas_hub.py              # 🚀 Main evaluation script
│
├── datasets/
├── Conjunto_Simplificaciones/             
│
├── AlignScore_v2_es/            # AlignScore implementation
├── QuestEval/                   # QuestEval implementation
├── moverscore/                  # MoverScore implementation
├── summac/                      # SummaC implementation
│
├── outputs/
│
├── README.md            
├── LICENSE
├── .gitignore
└── .gitattributes                                                                                                                                                                                                            
```
⚠️ Datasets and generated outputs are intentionally excluded.

## 🔧 Installation
- Clone the repository
```text
  git clone https://github.com/hulat-group/bilingual_simplification_metrics.git
  cd bilingual_simplification_metrics
```
- Create virtual environment
```text
  python3 -m venv metricas_env
  source metricas_env/bin/activate
```
- Install dependencies
```text
  pip install --upgrade pip
  pip install -r requirements.txt
```

## ⚠️ Important Notes
**English AlignScore requires checkpoint (English)**
- Manual download
1. Download from:
   https://huggingface.co/MASE98/alignscore-base-checkpoint/tree/main

2. Place the file at:

AlignScore_v2_es/checkpoint/checkpoints/AlignScore-base.ckpt
  
## 📂 Expected Data Structure
The evaluation pipeline assumes the following external directory structure. The original directory and references are expected to have files.
- **Spanish**/**English**
```text
datasets/
├── es/
│   ├── originals_txt_test/
│   └── references_txt_test/
│       ├── reference_1/
│       ├── reference_2/
│       └── ...
└── en/
    ├── originals_txt_test/
    └── references_txt_test/
        ├── reference_1/
        ├── reference_2/
        └── .../

```

## 🤖 Model-Generated Simplifications
```text
Conjunto_Simplificaciones/
├── es/
│   ├── model_1/
│   ├── model_2/
│   └── ...
└── en/
    ├── model_1/
    ├── model_2/
    └── ...
```
Each model folder must contain .txt files with the same basename as the corresponding original documents.

## 🚀 Running the Evaluation
- **Spanish**
```text
python metricas_hub.py \
  --lang es \
  --data_root ./datasets/es \
  --simpl_root ./Conjunto_Simplificaciones \
  --out ./outputs/es
```
- **English**
```text
python metricas_hub.py \
  --lang en \
  --data_root ./datasets/en \
  --simpl_root ./Conjunto_Simplificaciones \
  --out ./outputs/en
```

## 📊 Output Files
For each evaluated model:
```text
outputs/
├── metrics_<model>.json
├── metrics_<model>.csv
├── emissions_metricas_<model>.csv
└── emissions_metricas_GLOBAL.csv
```
