import os
import csv
import yaml
import openai


def read_config():
    with open("config.yaml", "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        return data[0]['key']

def write_config():
    api_info = [
        {
            'key': 'OPENAI_API_KEY'
        }
    ]

    with open("config.yaml", 'w') as yamlfile:
        data = yaml.dump(api_info, yamlfile)

# Read the specified prompt file (.csv)
# Column 0: Base prompt w/ no COT
# Column 1: COT-enhanced prompt
def read_prompts(fname):
    base_prompt_dir = "prompts"
    if not fname.endswith('.csv'):
        fname += '.csv'
    prompt_path = os.path.join(base_prompt_dir, fname)
    with open(prompt_path) as csvfile:
        reader = csv.reader(csvfile)
        header = True
        for row in reader:
            # Skip over the header line
            if header:
                header = False
                continue            
            basic_prompt = row[0]
            #print(basic_prompt)
            chain = row[1]
            #print(chain)

# Read the specified prompt from the prompt file (.csv)
# Column 0: Base prompt w/ no COT
# Column 1: COT-enhanced prompt
def read_prompt(fname, idx):
    base_prompt_dir = "prompts"
    if not fname.endswith('.csv'):
        fname += '.csv'
    prompt_path = os.path.join(base_prompt_dir, fname)
    with open(prompt_path) as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        idx = idx + 1 # skip header
        return rows[idx]
    

def run_codex(api_key, des_prompt):
    openai.api_key = api_key
    #des_prompt="#Write a python code for minimum deletes needed to make a string palindrome."
    #des_prompt+="Make the code executable, formatted and optimized\n\n"
    response = openai.Completion.create (engine="code-davinci-002",
                    prompt=des_prompt,
                    temperature=0.3,
                    max_tokens=512,
                    top_p=1,
                    frequency_penalty=0.7,
                    presence_penalty=0
                    )

    return response.choices[0].text

if __name__ == '__main__':
    api_key = read_config()
    #print(api_key)
    test_prompt = read_prompt('dp_probs1.csv', 0)
    #print(test_prompt)
    basic_prompt = test_prompt[0]
    cot_prompt = test_prompt[0] + test_prompt[1]
    codex_out = run_codex(api_key, basic_prompt)
    print(codex_out)
    