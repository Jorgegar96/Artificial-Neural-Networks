import numpy as np
import pandas as pd
from ArtificialNeuralNetwork import ArtificialNeuralNetwork

'''
data = pd.read_csv('./Datasets/part3_data_train.csv')
X = data.loc[:, data.columns != "clase"].to_numpy()
Y = pd.get_dummies(data["clase"]).to_numpy()
dimensions = np.array([len(X[0]),64, 5])
nn = ArtificialNeuralNetwork(dimensions)
nn.printNetwork()

nn.trainNetwork(X.T, Y.T)

val = pd.read_csv('./Datasets/part3_data_val.csv')
val_x = val.loc[:, val.columns != 'clase'].to_numpy()
val_y = pd.get_dummies(val["clase"]).to_numpy()
nn.test(val_x.T, val_y.T)

nn.test(X.T, Y.T)
'''
nn = ArtificialNeuralNetwork(jsonPath='./Datasets/part1_red_prueba.json')
nn.printNetwork()
