from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

X, y = load_wine(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

df = pd.DataFrame(X)
