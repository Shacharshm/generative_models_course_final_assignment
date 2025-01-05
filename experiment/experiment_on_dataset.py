import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from glob import glob

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

        ###### Pipeline here! #####

        df.loc[index, 'Output'] = "blabla" #output


    # save results
    df.to_csv(os.path.join(results_dir,results_file_name))

