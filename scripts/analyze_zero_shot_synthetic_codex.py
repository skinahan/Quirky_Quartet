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

    depths = [str(i) for i in range(1, 5)]
    ax = sns.lineplot(composite, x='depth', y='accuracy', errorbar='ci', hue='prompt')
    ax.set(title='Synthetic Dataset Prompting Comparison',
           xlabel='Steps', ylabel='Accuracy',
           xticks=range(1, 5), xticklabels=depths)
    sns.lineplot(x=[1, 2, 3, 4], y=[0.24, 0.07, 0.05, 0.02],
                 dashes=True, label='Codex Baseline') # Codex data

    save('plots/synthetic_prompt_evaluation.png', show=True)

    sns.set_color_codes('pastel')
    errors = composite[composite['behavior'] != 'Success']
    x = 'depth'; y = 'behavior'

    fig, axes = plt.subplots(1, 2)
    fig.suptitle('Error Analysis by Program Depth')
    for ax in axes:
        (errors.groupby(x)[y]
         .value_counts()
         .rename('count')
         .reset_index()
         .pipe((sns.barplot, 'data'), ax=ax, x=x, y='count', hue=y))
        ax.set(xlabel='Depth', ylabel='Count', xticklabels=depths[1:])
    save('plots/synthetic_error_analysis.png', show=True)
