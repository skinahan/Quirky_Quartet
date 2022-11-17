from library.openai_synthetic_dataset import generate

def run():
    for depth in range(1, 6):
        count = 0
        for program in generate(depth=depth):
            count += 1
        print(f'A depth of {depth} gives {count} programs (including duplicates)')
