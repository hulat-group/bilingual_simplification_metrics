import os
DIR = os.path.dirname(__file__)
__version__ = "0.2.4"

# Hacer que QuestEval sea importable directamente desde el paquete
from .questeval_metric import QuestEval

__all__ = ['QuestEval', '__version__', 'DIR']