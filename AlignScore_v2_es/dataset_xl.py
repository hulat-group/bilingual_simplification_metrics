# from datasets import load_dataset

# Cargar el dataset completo (en todos los idiomas)
# xnli_es = load_dataset("xnli", "es")


# Mostrar cuántos ejemplos hay en cada conjunto
# for split in xnli_es:
#     print(f"{split}: {len(xnli_es[split])} ejemplos")

# Mostrar el primer ejemplo del conjunto de entrenamiento
# print(xnli_es["train"][0])


from datasets import load_dataset
import json

# Cargar el dataset XNLI en español
xnli = load_dataset("xnli", "es")

# Map labels numéricos
label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

# Exportar función
def export_to_jsonl(split_name, split_data):
    output_path = f"xnli_es_{split_name}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in split_data:
            example = {
                "text_a": ex["premise"].strip(),
                "text_b": [ex["hypothesis"].strip()],
                "orig_label": ex["label"],
                "task": "nli"
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"✅ Exportado: {output_path} ({len(split_data)} ejemplos)")

# Exportar los tres splits
for split in ["train", "validation", "test"]:
    export_to_jsonl(split, xnli[split])
