from library.openai_synthetic_dataset import generate
import csv
from pathlib import Path

headers = ['id', 'description', 'code', 'outputs']

def run():
    for depth in range(1, 3):
        count = 0
        with open(f'data/synthetic_depth_{depth}.csv', 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(headers)
            for program in generate(depth=depth):
                count += 1
                print(program)
                writer.writerow([count, program.description, '', ''])
            print(f'A depth of {depth} gives {count} programs (including duplicates)')
