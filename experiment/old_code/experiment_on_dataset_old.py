import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from glob import glob

from huggingface_hub import login
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_guidance import LLMPipeline

# create result folder
base_dir = Path(__file__).parent.absolute().parent
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dir = os.path.join(base_dir, "results", timestamp)
os.makedirs(results_dir)

# get all csv path
dataset_dir = os.path.join(base_dir,"dataset")
all_csv_files = [file
                 for path, subdir, files in os.walk(dataset_dir)
                 for file in glob(os.path.join(path, "*.csv"))]



# login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a model and tokenizer
model_name = "gpt2" #"EleutherAI/gpt-neo-2.7B"
if device.type == "cpu":
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
elif device.type == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        use_auth_token=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=True
    )

model = model.to(device)

# Initialize the pipeline
llm_pipeline = LLMPipeline(model=model, tokenizer=tokenizer)


for category in all_csv_files:
    # print(category)
    csv_file_to_run=category

    results_file_name= os.path.basename(csv_file_to_run)
    df = pd.read_csv(csv_file_to_run, header=None)
    df.rename(columns={0: 'Prompt'}, inplace=True)
    df["Output"]=''
    df["Score"]=''

    for index, row in df.iterrows():
        prompt=row['Prompt']

        output = llm_pipeline(prompt, 
                          max_length=100, 
                          temperature=0.9,
                          top_k=50,
                          top_p=0.6,
                          do_sample=True,
                          use_cache=True,
                          repetition_penalty=1.0,
                          length_penalty=1.0,
                          stopping_criteria=None,
                          logits_processor=None,
                          guidance_scale=0.0)

        df.loc[index, 'Output'] = output


    # save results
    df.to_csv(os.path.join(results_dir,results_file_name))

