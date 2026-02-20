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
## 🪄 Simplification Quality
- **SARI** (ADD / KEEP / DELETE)

## 🔁 Semantic Similarity
- **BERTScore**
- **MoverScore**
- **Semantic Answer Similarity (SAS)**
- **BLEU**
- **ROUGE**

## 🧠 Factual Consistency
- **AlignScore**
- **SummaC** (ZS + Conv)
- **QuestEval**

## 📚 Readability
- Spanish → **Fernández-Huerta**
- English → **Flesch Reading Ease**

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
   https://huggingface.co/<your-username>/alignscore-base-checkpoint

2. Place the file at:

AlignScore_v2_es/checkpoint/checkpoints/AlignScore-base.ckpt
  
## 📂 Expected Data Structure
The evaluation pipeline assumes the following external directory structure.
- **Spanish**
```text
datasets/es/
├── originales_txt_test/
└── adaptaciones_txt_test/
    ├── FundacionAmas/
    ├── OscarGarciaMunoz/
    └── PlenaInclusionMadrid/
```
- **English**
```text
datasets/en/
├── originales_txt_test/
└── references_txt_test/
    ├── FundacionAmas/
    ├── OscarGarciaMunoz/
    └── PlenaInclusionMadrid/
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
