import csv
import ast
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run():
    dfs = []
    for depth in range(1, 5):
        print(f'Results for synthetic dataset, depth {depth}')
        df = pd.read_csv(f'results/synthetic_depth_{depth}.csv')
        extended = df[df['extended']]
        print('extended')
        print(extended.describe())
        normal = df[df['extended'] == False]
        print('normal')
        print(normal.describe())
        df['depth'] = depth
        dfs.append(df)
    composite = pd.concat(dfs)
    sns.lineplot(composite, x='depth', y='accuracy', errorbar='ci', hue='extended')
    plt.show()
