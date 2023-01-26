import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from collections import Counter
X, y = make_classification(n_samples=5000, n_features=40, n_informative=10,
                           n_redundant=30, weights=[0.91], random_state=1)
print(X)
print(y)
print(Counter(y))  # Counter({0: 4534, 1: 466})
print(X.shape)  # (5000, 40)
from imblearn.over_sampling import SMOTE
X1, y1 = SMOTE().fit_resample(X, y)
print(Counter(y1))  # Counter({0: 4534, 1: 4534})

from imblearn.under_sampling import RandomUnderSampler
under = RandomUnderSampler()
X2, y2 = under.fit_resample(X, y)
print(Counter(y2))  # Counter({0: 466, 1: 466})

from imblearn.pipeline import Pipeline
pipe = Pipeline([("Over", SMOTE(sampling_strategy=0.2)),
                 ("Under", RandomUnderSampler(sampling_strategy=0.5))])
X, y = pipe.fit_resample(X, y)
print(Counter(y)) # Counter({0: 1812, 1: 906})
