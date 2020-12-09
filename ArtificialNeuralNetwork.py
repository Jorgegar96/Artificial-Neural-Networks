import numpy as np
import pandas as pd
import json
from sklearn.metrics import mean_squared_error as MSE
from NetworkLayer import NetworkLayer
# from openpyxl import load_workbook  #Openpyxl needed


class ArtificialNeuralNetwork:

    # Dimensions: Corresponds to a list containing the total number of neurons per layer, including
    # input, hidden and output
    def __init__(self, dimensions=None, jsonPath=None, name="Test"):
        self.layers = {}
        self.best_weights = None  # JSON file containing a snapshot of the best weights
        self.name = name
        if jsonPath is None:
            self.initializeNetwork(dimensions)
        else:
            self.createFromJSON(jsonPath)

    def initializeNetwork(self, dimensions):
        for i, dimension in enumerate(dimensions):
            if i != len(dimensions)-1:
                self.layers[f"L{i}"] = NetworkLayer(
                    dimensions[i + 1], dimensions[i], act_func=lambda x: 1 / (1 + np.exp(-x))
                )
                continue
            break

    def trainNetwork(
            self, training_data, max_epochs, val_data=None, alpha=0.1, batch_size=1, max_nondecreasing=None, epsilon=None
    ):
        X = training_data.loc[:, training_data.columns != "clase"].to_numpy().T
        labels = pd.get_dummies(training_data["clase"]).to_numpy().T
        if val_data is not None:
            X_val = val_data.loc[:, val_data.columns != 'clase'].to_numpy().T
            val_labels = val_data.loc[:, 'clase'].to_numpy().T
            print(("{:<25}" * 5).format("Epoch", "Train Error", "Train MSE", "Test Error", "Test MSE"))
        else:
            print(("{:<25}" * 3).format("Epoch", "Train Error", "Train MSE"))
        train_error, train_mse = [], []
        test_error, test_mse, nondecr, min_error = [], [], 0, np.inf
        for i in range(max_epochs):
            for j in range(int(X.shape[1] / batch_size)):
                batch_data = np.array(X)[:, batch_size*j:batch_size*(j+1)]
                batch_labels = np.array(labels)[:, batch_size*j:batch_size*(j+1)]
                self.feedForward(batch_data)
                self.backPropagation(batch_data, batch_labels, alpha=alpha)
            if max_nondecreasing is not None and val_data is not None:
                errors = self.networkError(X, labels, i, X_val, val_labels, nondecr, min_error, epsilon)
                train_error.append(errors[0])
                train_mse.append(errors[1])
                test_error.append(errors[2])
                test_mse.append(errors[3])
                nondecr = errors[4]
                min_error = errors[5]
                if nondecr == max_epochs:
                    self.createFromJSON(self.best_weights)
                    break
                continue
            errors = self.networkError(X, labels, i)
            train_error.append(errors[0])
            train_mse.append(errors[1])

    def networkError(
            self, train_data, train_labels, index, test_data=None,
            test_labels=None, nondecr=0, min_error=None, epsilon=None
    ):
        train_error = self.calcError(self.feedForward(train_data), train_labels)
        train_mse = MSE(y_true=train_labels, y_pred=self.feedForward(train_data))
        errors = [train_error, train_mse]
        if test_data is not None and test_labels is not None:
            test_error = self.calcError(self.feedForward(test_data), test_labels)
            test_mse = MSE(y_true=test_labels, y_pred=self.feedForward(test_data))
            errors = [test_error, test_mse]
            min_error = min(min_error, test_error)
            condition = test_error[index] >= min_error or np.abs(test_error[index] - min_error) < epsilon
            nondecr = nondecr+1 if condition else 0
            self.best_weights = self.convertToJSON() if test_error[index] < min_error else self.best_weights
        r_format = "{:<25}" * (len(errors) + 1)
        print(r_format.format(index, *errors))
        errors.append(nondecr)
        errors.append(min_error)
        return errors

    def feedForward(self, entry_data):
        res = entry_data
        for key in self.layers.keys():
            res = np.dot(np.transpose(self.layers[key].weights).T, res)
            bias = self.layers[key].bias
            res += np.reshape(bias, (len(bias), 1))
            res = self.layers[key].act_func(res)
            self.layers[key].activations = res

        return res

    def backPropagation(self, entry_data, real_y, alpha=0.1):
        m = entry_data.shape[1]
        n_lay = len(self.layers) - 1
        delta = {}
        dZ = self.layers[f"L{n_lay}"].activations - real_y
        delta[f"dW{n_lay}"] = 1/m * np.dot(dZ, self.layers[f"L{n_lay-1}"].activations.T)
        delta[f"db{n_lay}"] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        delta[f"dZ{n_lay}"] = dZ
        n_lay -= 1

        #Delta propagation
        while n_lay >= 0:
            term1 = 1/m * np.dot(self.layers[f"L{n_lay+1}"].weights.T, delta[f"dZ{n_lay+1}"])
            term2 = np.multiply(self.layers[f"L{n_lay}"].activations, (1-self.layers[f"L{n_lay}"].activations))
            dZ = np.multiply(term1, term2)
            if n_lay != 0:
                activations = self.layers[f"L{n_lay-1}"].activations
            else:
                activations = entry_data
            delta[f"dW{n_lay}"] = 1/m * np.dot(dZ, activations.T)
            delta[f"db{n_lay}"] = 1/m * np.sum(dZ, axis=1, keepdims=True)
            delta[f"dZ{n_lay}"] = dZ
            n_lay -= 1

        #Weight update
        for i, key in enumerate(self.layers.keys()):
            self.layers[key].weights = self.layers[key].weights - alpha * delta[f"dW{i}"]
            self.layers[key].bias = self.layers[key].bias - alpha * delta[f"db{i}"]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def linear(self, x):
        return x

    def calcError(self, y_pred, y_true):
        error = -(np.multiply(y_true, np.log2(y_pred)) + np.multiply(1-y_true, np.log2(1 - y_pred)))
        return np.sum(np.sum(error))

    def printNetwork(self):
        for i, key in enumerate(self.layers):
            print(f"Layer {i}", end='')
            print("-"*100)
            n_elem = len(self.layers[key].weights) + 1
            r_format = "{:<25}" * n_elem
            print(r_format.format("", *[f"Neuron #{x}" for x in range(n_elem)]))
            values = np.transpose(self.layers[key].bias)
            for j in range(len(values)):
                print(r_format.format("Bias", *values[j]))
            values = np.transpose(self.layers[key].weights)
            for j in range(len(values)):
                print(r_format.format(f"Weight {j}", *values[j]))

    def predict(self, features):
        return self.feedForward(features)

    def test(self, X, Y):
        predictions = self.predict(X)
        actual = np.argmax(Y, axis=0)
        pred = np.argmax(predictions, axis=0)
        match = actual == pred
        print(np.sum(match)/len(match))

    def createFromJSON(self, jsonPath=None, jsonNet=None):
        if jsonNet is None and jsonPath is not None:
            with open(jsonPath) as file:
                network = json.load(file)
        elif jsonNet is not None and jsonPath is None:
            network = jsonNet
        else:
            print("Invalid call format")
            return None
        self.layers = {}
        for i, layer in enumerate(network['capas']):
            self.layers[f"L{i}"] = NetworkLayer(layer=layer, act_func=lambda x: 1 / (1 + np.exp(-x)))

    def saveAsJSON(self):
        network = self.convertToJSON()
        with open(f'./Networks/Network-{self.name}.json', 'w') as file:
            json.dump(network, file, indent=4)

    def convertToJSON(self):
        network = {"entradas": self.layers['L0'].weights.shape[1], "capas": []}
        for layer in self.layers:
            network['capas'].append({
                "neuronas": [
                    {"pesos": weight.tolist()} for weight in self.layers[layer].weights
                ]
            })
        return network

    def saveAsExcel(self):
        conf_route = f'./Networks/{self.name}.xlsx'
        writer = pd.ExcelWriter(conf_route, engine='openpyxl')
        for i, key in enumerate(self.layers):
            layer_dict = {}
            for j, neuron in enumerate(self.layers[key].bias):
                layer_dict[f'Neuron{j}'] = neuron
            for j, neuron in enumerate(self.layers[key].weights):
                layer_dict[f'Neuron{j}'] = np.append(layer_dict[f'Neuron{j}'], neuron)
            index = ['Bias'] + [f'Weight{x}' for x in range(len(self.layers[key].weights[0]))]
            layerdf = pd.DataFrame(data=layer_dict, index=index)
            layerdf.to_excel(excel_writer=writer, sheet_name=f'Layer{i}')
        writer.save()
        writer.close()
