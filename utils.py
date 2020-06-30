from os import listdir
from os.path import isfile, join
from datetime import datetime
from pathlib import Path
import pickle
import pandas as pd
import seaborn as sns


def plot_directory(path: str, env_name: str, directories: list, ci=None, estimator='mean'):
    all_records = []
    for d, name in directories:
        mean_rewards = load_version(f"{path}/{d}", env_name)  # dilemma, no agent
        flatten_rewards = sum(mean_rewards, [])
        records = [(r[0], r[1], name) for r in flatten_rewards]
        all_records.extend(records)
    data = pd.DataFrame.from_records(all_records, columns=['number of games', 'score', 'agent'])
    ax = sns.lineplot(x='number of games', y='score', data=data, hue='agent', ci=ci, estimator=estimator)
    return ax


def save_file(path: str, version: str, data):
    Path(path).mkdir(parents=True, exist_ok=True)
    with open(f"{path}/{version}_{_get_current_timestamp()}.pickle", 'wb') as file:
        pickle.dump(data, file)


def read_file(path: str, version: str):
    files = [file for file in listdir(path) if isfile(join(path, file)) and file.startswith(version)]
    last_file = max(files)
    with open(f"{path}/{last_file}", 'rb') as file:
        return pickle.load(file)


def load_version(path: str, version: str):
    files = [file for file in listdir(path) if isfile(join(path, file)) and file.startswith(version)]
    result = []
    for file_name in files:
        with open(f"{path}/{file_name}", 'rb') as file:
            result.append(pickle.load(file))
    return result


def _get_current_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
