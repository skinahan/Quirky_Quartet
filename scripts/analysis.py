import csv
import ast
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run():
    sns.set_style('whitegrid')
    dfs = []
    for depth in range(1, 5):
        print(f'Results for synthetic dataset, depth {depth}')
        df = pd.read_csv(f'results/zero_shot/synthetic_depth_{depth}.csv')
        extended = df[df['extended']]
        print('extended')
        print(extended.describe())
        normal = df[df['extended'] == False]
        print('normal')
        print(normal.describe())
        df['prompt'] = df['extended'].apply(lambda e : 'Compositional' if e else 'Default')
        df['depth'] = depth
        dfs.append(df)
    composite = pd.concat(dfs)
    ax = sns.lineplot(composite, x='depth', y='accuracy', errorbar='ci', hue='prompt')
    ax.set(title='Synthetic Dataset Prompting Comparison',
           xlabel='Steps', ylabel='Accuracy',
           xticks=range(1, 5), xticklabels=[str(i) for i in range(1, 5)])
    sns.lineplot(x=[1, 2, 3, 4], y=[0.24, 0.07, 0.05, 0.02],
                 dashes=True, label='Codex Baseline') # Codex data
    plt.savefig('synthetic_prompt_evaluation.png')
    plt.show()
