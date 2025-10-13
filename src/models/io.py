import pandas as pd


def read_split(proc_dir: str, split: str) -> pd.DataFrame:
    return pd.read_csv(f"{proc_dir}/{split}.csv")

def read_split_with_prompt(proc_dir: str, split: str) -> pd.DataFrame:
    return pd.read_csv(f"{proc_dir}/{split}_with_prompt.csv")
