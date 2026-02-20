# test_alignscore_es.py
from src.alignscore import AlignScore
from textos_cl import originales, adaptados

# Inicializa AlignScore con modelo en español
align = AlignScore(
    #model="google-bert/bert-base-multilingual-cased",  
    #model="dccuchile/bert-base-spanish-wwm-cased", 
    #batch_size=4,
    #device=0,  # o 'cuda' si tienes GPU, o 'cpu' para probar
    #ckpt_path="checkpoints_alignscore_es/dccuchile-bert-base-spanish-wwm-cased_xnli_es_500000_16x1x1_epoch=00_step=20000.ckpt",  # sin checkpoint entrenado
    #evaluation_mode='nli_sp'  # modo por defecto: clasificación de NLI soft
    model="dccuchile/bert-base-spanish-wwm-cased",
    batch_size=32,
    device='cuda:0',
    ckpt_path="checkpoints_alignscore_es/dccuchile-bert-base-spanish-wwm-cased_step=60000.ckpt",    
    #ckpt_path="checkpoints1_alignscore_es/dccuchile-bert-base-spanish-wwm-cased_final.ckpt",  
    evaluation_mode='nli_sp' 
)

# Calcula AlignScore
scores = align.score(originales, adaptados)

# Imprime resultados
for i, s in enumerate(scores):
    print(f"✅ Par {i+1} | AlignScore: {s:.4f}")
