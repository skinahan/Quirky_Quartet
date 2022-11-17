from library.openai_synthetic_dataset import generate, test_cases
import csv
from pathlib import Path
from rich import print

def run():
    headers = (['id', 'description']
               + list(sorted(test_cases.keys())))
    test_inputs = [v for k, v in sorted(test_cases.items(), key=lambda t:t[0])]

    for depth in range(1, 6):
        count = 0
        with open(f'data/synthetic_depth_{depth}.csv', 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(headers)
            for program in generate(depth=depth):
                count += 1
                print(program.description)
                print()
                outputs = [program.rule(test_input) for test_input in test_inputs]
                for output in outputs:
                    print(f'    "{output}"')
                print()
                writer.writerow([count, program.description] + outputs)
            print(f'A depth of {depth} gives {count} programs (including duplicates)')
