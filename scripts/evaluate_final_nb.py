"""Run only the frozen final NB configurations and the causal ten-year baseline.

python3.12 scripts/evaluate_final_nb.py
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lab1-matplotlib')
os.environ.setdefault('MPLBACKEND', 'Agg')

from src.nb_final import run_final_evaluation


if __name__ == '__main__':
    result = run_final_evaluation(ROOT)
    print(result['summary'].to_string(index=False))
    print(result['class_report'].to_string(index=False))
    print('Diferencias entre ambos NB:', result['diagnosis']['n_differences'])
    print('Resultados: results/naive_bayes/final/')
