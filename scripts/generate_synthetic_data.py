from library.openai_synthetic_dataset import generate, test_cases, format_source
import csv
from pathlib import Path
from rich import print
import hashlib

def run():
    headers = (['id', 'description', 'code', 'md5']
               + list(sorted(test_cases.keys())))
    test_inputs = [v for k, v in sorted(test_cases.items(), key=lambda t:t[0])]

    seen = set()

    for depth in range(1, 6):
        count = 0
        duplicated = 0
        with open(f'data/synthetic_depth_{depth}.csv', 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(headers)
            for program in generate(depth=depth):
                outputs = [program.rule(test_input) for test_input in test_inputs]
                combined = ''
                for output in outputs:
                    # print(f'    "{output}"')
                    combined += output
                h = hashlib.md5(combined.encode('utf-8')).hexdigest()
                if h not in seen:
                    print(program.description)
                    print(h)
                    count += 1
                    source = format_source(program)
                    print('Generated code:')
                    print(source)
                    writer.writerow([count, program.description, source, h] + outputs)
                    seen.add(h)
                else:
                    duplicated += 1
            print(f'A depth of {depth} gives {count} programs (excluding duplicates)')
            print(f'There were {duplicated} duplicates')
