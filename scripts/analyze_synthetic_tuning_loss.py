import csv
import ast
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from .analyze_zero_shot_synthetic_codex import save

# test_model in lib/t5_tuning

def run():
    sns.set_style('whitegrid')
    path = Path('results/csv_data/synthetic/version_0/metrics.csv')
    df = pd.read_csv(path)
    print(df)
    print(df.describe())
    sns.lineplot(data=df, x='epoch', y='training_loss')
    save('plots/synthetic_training_loss.png', show=True)
