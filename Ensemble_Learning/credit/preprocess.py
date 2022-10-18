import pandas as pd
import numpy as np

dataset = pd.read_csv('./UCI_Credit_Card.csv', header=None)

s = dataset.sample(frac=1, replace=False)
train_data = s.iloc[:24000]
test_data = s.iloc[24000:]

train_data.to_csv('./train.csv', header=None, index=False)
test_data.to_csv('./test.csv', header=None, index=False)