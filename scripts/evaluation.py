import human_eval
from human_eval.human_eval.data import write_jsonl, stream_jsonl, read_problems

def generate_one_completion(prompt):
    return 'blah'

def run():
    print('Evaluating')
    problems = read_problems()

    num_samples_per_task = 200
    samples = [
        dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
        for task_id in problems
        for _ in range(num_samples_per_task)
    ]
    filename = 'samples.jsonl'
    write_jsonl(filename, samples)
    for line in stream_jsonl(filename):
        print(line)
