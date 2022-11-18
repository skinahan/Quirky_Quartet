import pandas as pd

def run():
    for depth in range(1, 6):
        filename = f'data/synthetic_depth_{depth}.csv'
        df = pd.read_csv(filename)
        new_df = df.drop(columns=['whitespace'])
        new_df.to_csv(filename, index=False)
