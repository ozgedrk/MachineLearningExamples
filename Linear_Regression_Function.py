import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 

df = pd.read_csv("weight-height.csv")
df.head()
df.info()

y = df["Kilo"]
X = df["Boy"]
X.head()