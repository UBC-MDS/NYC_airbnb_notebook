# author: Revathy
# date: 2022-12-10
"""

"Cleans, splits and pre-processes the airbnb data (from https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data).

Writes the training and test data to separate csv files.
Usage: src/01-preprocess_airbnb.py --input_file=<input_file> --out_dir=<out_dir>

Example script to run in terminal: 

python src/01-preprocess_airbnb.py --input_file="data/raw/AB_NYC_2019.csv" --out_dir="data/processed"

  
Options:
--input_file=<input_file>       Path (including filename) to raw data (csv file)
--out_dir=<out_dir>   Path to directory where the processed data should be written
"
"""

# import required libraries
from docopt import docopt
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#initialize docopt
dc = docopt(__doc__)


#function to read and clean the data 
def main(input_file, out_dir):

    # read data and convert 
    data_raw = pd.read_csv('data/AB_NYC_2019.csv', parse_dates=['last_review'])
    
    #all the rows will na values are dropped 
    data = data_raw.dropna().reset_index().drop(columns=['index'])
    
    
    # split data into training and test sets
    train_df, test_df = train_test_split(data, test_size=0.3, random_state=123)
    
    # write training and test data to csv files
    try:
        train_df.to_csv(f'{out_dir}/train.csv', index=False)
        test_df.to_csv(f'{out_dir}/test.csv', index=False)
    except:
        os.makedirs(os.path.dirname('data/processed/'))
        train_df.to_csv(f'{out_dir}/train.csv', index=False)
        test_df.to_csv(f'{out_dir}/test.csv', index=False)

if __name__ == "__main__":
    main(dc["--input_file"], dc["--out_dir"])
