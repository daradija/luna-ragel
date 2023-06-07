from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url="https://github.com/ageron/data/raw/main/housing.tgz"
        print("Downloading housing data...")
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tgz:
            housing_tgz.extractall(path="datasets")
    return pd.read_csv("datasets/housing/housing.csv")

def printData(housing):
    import matplotlib.pyplot as plt
    housing.hist(bins=50, figsize=(12,8))
    plt.show()

def shuffle_and_split_data(data, test_ratio):
    import numpy as np
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

housing=load_housing_data()
train_set, test_set = shuffle_and_split_data(housing, 0.2)
#printData(data)
print(len(train_set), "train +", len(test_set), "test")