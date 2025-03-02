# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 12:56:36 2025

@author: vishaladithyaa
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_csv_filepath = os.path.join("Data","training.csv")

train_df = pd.read_csv(train_csv_filepath)

train_df.fillna(train_df.mean(),inplace = True)
train_df.dropna(how = "all")

train_df["Image"] = train_df["Image"].apply(lambda x: np.fromstring(x, sep = " ").reshape(96,96,1))

X_imgs = np.stack(train_df["Image"].values) / 255.0
y = train_df.drop(columns = ["Image"]).values

plt.figure(figsize=(4, 4))
plt.imshow(X_imgs[1], cmap="gray")
plt.title("96x96")
plt.grid(True)
plt.show()