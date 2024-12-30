# Time-sorting assistance imports
import datetime
import time
import calendar

# For Data (map) visualization
from mpl_toolkits.basemap import Basemap

# Machine Learning Part
from sklearn.model_selection import train_test_split

# Neural Networks
# import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV

# Scale your input and output data
from sklearn.preprocessing import StandardScaler

# Data analysis inports
import numpy as np
import pandas as pd
import matplotlib.pyplot as mlp

# Load datatset using pandas
data = pd.read_csv("data.csv")
# print(data.columns)

# Filter to categorize that impact EQ data
data = data[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]
# print(data.head())

timestamp = []


def get_timestamp(tp):
    # Calculate days since epoch: "Jan 1, 1970"
    dt = datetime.datetime(tp.tm_year, tp.tm_mon, tp.tm_mday,
                           tp.tm_hour, tp.tm_min, tp.tm_sec)
    epoch = datetime.datetime(1970, 1, 1)

    # Convert difference to seconds
    delta = dt - epoch
    return delta.total_seconds()


for tim, date in zip(data['Time'], data['Date']):
    try:
        timestamp_ = datetime.datetime.strptime(date + ' ' + tim,
                                                '%m/%d/%Y %H:%M:%S')
        tp = timestamp_.timetuple()

        timestamp.append(get_timestamp(tp))
    except ValueError:
        timestamp.append("Error")

timeStamp = pd.Series(timestamp)
data['Timestamp'] = timeStamp.values
final_data = data.drop(['Date', 'Time'], axis=1)
final_data = final_data[final_data.Timestamp != "Error"]
# print(data.head())

map = Basemap(projection='mill', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180,
              urcrnrlon=180, lat_ts=20, resolution='c')

lat = data["Latitude"].tolist()
long = data["Longitude"].tolist()

x, y = map(long, lat)
fig = mlp.figure(figsize=(12, 10))

mlp.title("All affected areas")
map.plot(x, y, "o", markersize=2, color='blue')

map.drawcoastlines()
map.fillcontinents(color='coral', lake_color='aqua')
map.drawmapboundary()
map.drawcountries()

# Print map
# mlp.show()

X = final_data[['Timestamp', 'Latitude', 'Longitude']]
Y = final_data[['Magnitude', 'Depth']]

X_tr, X_tst, Y_trn, Y_tst = train_test_split(X, Y, test_size=0.2,
                                             random_state=42)


# Train and Test data check
# print(X_tr.shape, X_tst.shape, Y_trn.shape, Y_tst.shape)

def create_model(hidden_layer_sizes=(16,), activation='relu', optimizer='adam', loss='mse'):
    model = Sequential()
    model.add(Dense(hidden_layer_sizes[0], activation=activation, input_shape=(3,)))
    model.add(Dense(hidden_layer_sizes[0], activation=activation))
    # Change final layer activation since this is a regression problem
    model.add(
        Dense(2, activation='softmax'))  # Changed from 'softmax' to 'linear'
    model.compile(optimizer, loss, metrics=['accuracy'])

    return model

# Create the KerasRegressor instead of KerasClassifier
model = KerasRegressor(
    model=create_model,
    hidden_layer_sizes=(16,),
    verbose=1
)

# Define parameter grid
param_grid = {
    'batch_size': [30],  # Increased batch size
    'epochs': [10],
    'model__optimizer': ['Adamax', 'Adam'],
    'model__loss': ['squared_hinge'],  # better than mse
    'model__activation': ['relu', 'sigmoid']
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    n_jobs=-1)

# Scale inputs
scaler_X = StandardScaler()
X_tr_scaled = scaler_X.fit_transform(X_tr)
X_tst_scaled = scaler_X.transform(X_tst)

# Scale outputs
scaler_Y = StandardScaler()
Y_trn_scaled = scaler_Y.fit_transform(Y_trn)
Y_tst_scaled = scaler_Y.transform(Y_tst)

# Fit the model with scaled data
grid_result = grid.fit(X_tr_scaled, Y_trn_scaled)

# grid_result = grid.fit(X_tr, Y_trn)


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


model = Sequential()
model.add(Dense(16, activation='linear', input_shape=(3,)))
model.add(Dense(16, activation='linear'))
model.add(Dense(2, activation='softmax'))

X_tr = X_tr.apply(pd.to_numeric, errors='coerce').fillna(0).values
Y_trn = Y_trn.apply(pd.to_numeric, errors='coerce').fillna(0).values
X_tst = X_tst.apply(pd.to_numeric, errors='coerce').fillna(0).values
Y_tst = Y_tst.apply(pd.to_numeric, errors='coerce').fillna(0).values


# print(X_tr.head(), Y_trn.head())

# Model and parameters chosen as per best result highlighted in line 144
model.compile(optimizer='Adamax', loss='squared_hinge', metrics=['accuracy'])
model.fit(X_tr, Y_trn, batch_size=30, epochs=10, verbose=1, validation_data=(
    X_tst, Y_tst))

[test_loss, test_acc] = model.evaluate(X_tst, Y_tst)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format
      (test_loss, test_acc))
