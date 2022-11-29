import csv
import ast
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def save(path, show=True):
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

def run():
    df = pd.read_excel('results/t5_tuning_results.xlsx')
    print(df.head(100))
    # print(df.describe())
