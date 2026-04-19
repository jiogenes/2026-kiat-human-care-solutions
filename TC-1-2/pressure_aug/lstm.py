import os
import warnings
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

Data_Path = './ventilator-pressure-prediction/'

df_train = pd.read_csv(Data_Path+'train.csv')
df_test = pd.read_csv(Data_Path+'test.csv')

df_sample = df_train[df_train['breath_id']<5].reset_index(drop=True)

print(df_train.head())
print(df_train.info())