#import packages
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import pickle

#upload dataset
url = 'https://raw.githubusercontent.com/samkrall/model/main/data/data_file'
df = pd.read_csv(url, index_col = 0)

#build model
X = df.drop('heart.disease', axis=1)
y = df['heart.disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.7, random_state = 69)

my_model = linear_model.LinearRegression()

my_model.fit(X_train, y_train)

#pickle model
pickle.dump(my_model, open('model.pk1', 'wb'))

my_model = pickle.load(open('model.pk1', 'rb'))
