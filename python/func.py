#place to store re-usable methods
import pandas as pd

def load_data():
    file_path = "/Users/pengchengliu/go/src/github.com/user/Titanic/data/"
    train = pd.read_csv(file_path+"train.csv")
    test = pd.read_csv(file_path+"test.csv")
    return train, test