# 🌍 Bilingual Simplification Metrics  
### 🧠 A Unified Evaluation Framework for Text Simplification (ES/EN)

Robust and reproducible framework for the automatic evaluation of **text simplification systems** in Spanish and English.

This repository integrates structural simplification metrics, semantic similarity measures, factual consistency evaluation, readability analysis, and environmental impact tracking into a single unified pipeline.

---

## ✨ Key Features

🔹 **Bilingual support** (Spanish 🇪🇸 / English 🇬🇧)  
🔹 Structured evaluation across multiple models  
🔹 JSON + CSV export (publication-ready)  
🔹 CodeCarbon integration (sustainability tracking)  
🔹 Modular metric design  

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
├── AlignScore_v2_es/            # AlignScore implementation
├── QuestEval/                   # QuestEval implementation
├── moverscore/                  # MoverScore implementation
├── summac/                      # SummaC implementation
│
├── README.md            
├── LICENSE
├── .gitignore
└── .gitattributes                                                                                                                                                                                                            
