from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




sample = pd.read_csv("sample_sub_reg.csv")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

def datachecks(df):
    return("HEAD and TAIL: ",df.head(3), df.tail(3))
    return("NULL VALUES: ",df.isnull().sum(),"IS NA: ", df.isna().sum().sum())
    return("DESCRIPTIONS: ", df.describe().T, df.info())
    
datachecks(train)

