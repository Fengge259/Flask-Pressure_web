import pandas as pd
import numpy as np
import pickle


df = pd.read_csv('Pressure_1.csv')

X = np.array(df.iloc[:, 0:4])
y = np.array(df.iloc[:, 4:])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# from sklearn.svm import SVC
# sv = SVC(kernel='linear').fit(X_train, y_train)

from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(max_depth=2, random_state=0)
sv = regr.fit(X, y)


pickle.dump(sv, open('pre.pkl', 'wb'))