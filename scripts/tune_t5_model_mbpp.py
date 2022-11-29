from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from csv import writer

from library.t5_model import CodeT5
from library.mbpp import *
from transformers import T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup


# Contents adapted from:
# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/T5/Fine_tune_CodeT5_for_generating_docstrings_from_Ruby_code.ipynb

def preprocess_examples(examples):
    # encode the code-docstring pairs
    codes = examples['code']
    docstrings = examples['text']
    # [SK] note: additional prompting methods can be introduced here
    prefix = "Generate Python: "
    max_input_length = 512
    max_target_length = 256
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
    inputs = [prefix + docstring for docstring in docstrings]
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

    # encode the code responses
    labels = tokenizer(codes, max_length=max_target_length, padding="max_length", truncation=True).input_ids

    # important: we need to replace the index of the padding tokens by -100
    # such that they are not taken into account by the CrossEntropyLoss
    labels_with_ignore_index = []
    for labels_example in labels:
        labels_example = [label if label != 0 else -100 for label in labels_example]
        labels_with_ignore_index.append(labels_example)

    model_inputs["labels"] = labels_with_ignore_index

    return model_inputs


def tune_model():
    dataset = load_dataset("mbpp")
    print(dataset)

    example = dataset['train'][0]

    print("Code:", example["code"])
    print("Docstring:", example["text"])
    dataset = dataset.map(preprocess_examples, batched=True)
    dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
    train_dataloader = DataLoader(dataset['train'], shuffle=True, batch_size=8, num_workers=4)
    valid_dataloader = DataLoader(dataset['validation'], batch_size=4, num_workers=4)
    test_dataloader = DataLoader(dataset['test'], batch_size=4, num_workers=4)
    batch = next(iter(train_dataloader))
    print(batch.keys())
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
    tokenizer.decode(batch['input_ids'][0])
    labels = batch['labels'][0]
    tokenizer.decode([label for label in labels if label != -100])
    model = CodeT5(train_dataloader, valid_dataloader, test_dataloader)
    early_stop_callback = EarlyStopping(
        monitor='validation_loss',
        patience=3,
        strict=False,
        verbose=False,
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    use_gpu = True
    if use_gpu:
        trainer = Trainer(gpus=1,
                          default_root_dir="./Checkpoints",
                          callbacks=[early_stop_callback, lr_monitor])
    else:
        trainer = Trainer(['cpu'],
                          default_root_dir="./Checkpoints",
                          callbacks=[early_stop_callback, lr_monitor])
    trainer.fit(model)
    # save in the current working directory, you can change this of course
    save_directory = "./"
    model.model.save_pretrained(save_directory)


def test_model():
    save_file = "./"
    model = T5ForConditionalGeneration.from_pretrained(save_file)
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
    dataset = load_dataset("mbpp")
    test_example = dataset['test'][2]
    # print("Code:", test_example['code'])
    # prepare for the model
    input_ids = tokenizer(test_example['text'], return_tensors='pt').input_ids
    # generate
    outputs = model.generate(input_ids)
    # print("Generated code:", tokenizer.decode(outputs[0], skip_special_tokens=False))
    # print("Ground truth:", test_example['code'])
    test_set = dataset['test']

    headers = ['docstring', 'label', 'output']
    with open('t5_out.csv', 'a', encoding="utf-8") as f_obj:
        writer_obj = writer(f_obj)
        writer_obj.writerow(headers)
        for val in test_set:
            row_list = []
            docstring = val['text']
            label = val['code']
            input_ids = tokenizer(val['text'], return_tensors='pt').input_ids
            # Using default generation setting: greedy decoding
            outputs = model.generate(input_ids, max_length=len(val['code']))
            outputs = [tokenizer.decode(x, skip_special_tokens=True) for x in outputs]
            decoded = ''.join(outputs)
            row_list.append(docstring)
            row_list.append(label)
            row_list.append(decoded)
            writer_obj.writerow(row_list)
        f_obj.close()


def eval_instance(instance, query, t5_out):
    verbose = 1
    tests = [parse_test_asserts(ast.parse(t)) for t in instance["test_list"]]
    try:
        status, code = eval_codex_out(t5_out, tests, verbose=verbose)
    except:
        status = 1
        code = t5_out
    if status == 0:
        if verbose > 0:
            print("Success")
    return status, query, t5_out, code


def eval_model(tuned=False):
    dataset = load_dataset("mbpp")
    test_set = dataset['test']
    results = []
    status = []
    csv_file = './results/t5/mbpp_t5/t5_out_mbpp_untuned.csv'
    if tuned:
        csv_file = './results/t5/mbpp_t5/t5_out_mbpp_tuned.csv'
    ground_truth = True

    with open(csv_file, 'r', encoding="utf-8") as f_obj:
        reader_obj = csv.reader(f_obj)
        idx = 0
        header_row = True
        max_len = len(test_set) - 1
        for row in reader_obj:
            if header_row:
                header_row = False
            else:
                if idx > max_len:
                    break
                query = row[0]
                t5_out = row[2]
                if ground_truth:
                    t5_out = row[1]  # Use the label instead of the generated code
                instance = test_set[idx]
                # Sanity check: Make sure the queries line up
                # print(query)
                # print(instance['text'])
                res = eval_instance(instance, query, t5_out)
                results.append(res)
                status.append(True if res[0] == 0 else False)
                idx = idx + 1
    return results, status


def eval_and_save(tuned=False):
    results, status = eval_model(tuned)
    # status = np.any(status)
    ground_truth = True
    result_dir = "./results/t5/mbpp_t5/eval_out/untuned"
    if tuned:
        result_dir = "./results/t5/mbpp_t5/eval_out/tuned"
    if ground_truth:
        result_dir = "./results/t5/mbpp_t5/eval_out/labels"
    os.makedirs(result_dir, exist_ok=True)
    for idx in range(len(results)):
        query = results[idx][1]
        # print(status)
        d = {"query": query, "success": 1 if status[idx] else 0}
        save_result_file(d, os.path.join(result_dir, "{0}.json".format(idx)), is_json=True, is_pickle=False)


def gen_summary(tuned=False):
    result_dir = "./results/t5/mbpp_t5/eval_out/untuned"
    if tuned:
        result_dir = "./results/t5/mbpp_t5/eval_out/tuned"
    total_files = 0
    total_success = 0
    for file in os.listdir(result_dir):
        if file.endswith('.json'):
            total_files = total_files + 1
            f = os.path.join(result_dir, file)
            if os.path.isfile(f):
                # load the json file
                f_obj = open(f)
                data = json.load(f_obj)
                f_obj.close()
                if data["success"] == 1:
                    total_success = total_success + 1
    print("Summary statistics:")
    if total_success > 0:
        pass_rate = total_success / total_files
    else:
        pass_rate = 0.0
    print(f"Pass Rate: {pass_rate}")


def run():
    # tune_model()
    # test_model()
    eval_and_save()
    # gen_summary(True)

