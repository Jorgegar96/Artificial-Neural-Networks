import numpy as np
import pandas as pd
import json
from sklearn.metrics import f1_score, classification_report, confusion_matrix, mean_squared_error as MSE
from NetworkLayer import NetworkLayer


# from openpyxl import load_workbook  #Openpyxl needed


def testAccuracy(y_pred, y_true):
    actual = np.argmax(y_true, axis=0)
    pred = np.argmax(y_pred, axis=0)
    match = actual == pred
    return np.sum(match) / len(match)


def testF1(y_pred, y_true):
    actual = np.argmax(y_true, axis=0)
    predictions = np.argmax(y_pred, axis=0)
    return f1_score(y_true=actual, y_pred=predictions, average='macro')


def testF1PerClass(y_pred, y_true, pos_class):
    actual = np.argmax(y_true, axis=0)
    predictions = np.argmax(y_pred, axis=0)
    return classification_report(y_true=actual, y_pred=predictions, output_dict=True)[f'{pos_class}']['f1-score']


def getConfusionMatrix(y_pred, y_true):
    actual = np.argmax(y_true, axis=0)
    predictions = np.argmax(y_pred, axis=0)
    return confusion_matrix(actual, predictions)


def sigmoid(x):
    if -x > 700:
        return 0.0
    return 1.0 / (1.0 + np.exp(-x))


class ArtificialNeuralNetwork:

    # Dimensions: Corresponds to a list containing the total number of neurons per layer, including
    # input, hidden and output
    def __init__(self, dimensions=None, jsonPath=None, name="Test", output_funct="Sigmoid"):
        self.layers = {}
        self.best_weights = None  # JSON file containing a snapshot of the best weights
        self.name = name
        if jsonPath is None:
            self.initializeNetwork(dimensions, output_funct)
        else:
            self.createFromJSON(jsonPath)

    def initializeNetwork(self, dimensions, output_func):
        act_func = np.vectorize(sigmoid)
        for i, dimension in enumerate(dimensions):
            if i != len(dimensions) - 2:
                self.layers[f"L{i}"] = NetworkLayer(
                    dimensions[i + 1], dimensions[i], act_func=act_func
                )
                continue
            elif i == len(dimensions) - 2:
                if output_func == "Linear":
                    act_func = lambda x: x
                self.layers[f"L{i}"] = NetworkLayer(
                    dimensions[i + 1], dimensions[i], act_func=act_func
                )
            break

    def trainNetwork(
            self, training_data, training_labels, max_epochs, val_data=None, val_labels=None,
            alpha=0.1, batch_size=1, max_nondecreasing=None, epsilon=None, output_type="Undetermined",
            error_metric="MSE"
    ):
        results = self.initResults(val_data, output_type)
        nondecr, min_error = 0, np.inf
        for i in range(max_epochs):
            for j in range(int(training_data.shape[1] / batch_size)):
                batch_data = np.array(training_data)[:, batch_size * j:batch_size * (j + 1)]
                batch_labels = np.array(training_labels)[:, batch_size * j:batch_size * (j + 1)]
                self.feedForward(batch_data)
                self.backPropagation(batch_data, batch_labels, alpha=alpha)
            if max_nondecreasing is not None and val_data is not None:
                nondecr, min_error = self.networkError(
                    training_data, training_labels, i, results, output_type,
                    val_data, val_labels, nondecr, min_error, epsilon, error_metric
                )
                if nondecr == max_nondecreasing:
                    self.createFromJSON(jsonNet=self.best_weights)
                    break
                continue
            self.networkError(training_data, training_labels, i, results, output_type)
        return results

    def initResults(self, val_data, output_type):
        results = {'train_mse': []}
        headers = ["Epoch", "Train MSE"]
        if output_type != 'Regression':
            results['train_error'] = []
            headers += ["Train CV Error"]
        if output_type == 'Classification':
            results['train_f1'] = []
            results['train_acc'] = []
            headers += ["Train F1", "Train Accuracy"]
        if val_data is not None:
            results['val_mse'] = []
            headers += ["Validation MSE"]
            if output_type != 'Regression':
                results['val_error'] = []
                headers += ["Validation CV Error"]
            if output_type == 'Classification':
                results['val_f1'] = []
                results['val_acc'] = []
                headers += ["Validation F1", "Validation Accuracy"]
        print(("{:<8}" + "{:<23}" * (len(headers) - 1)).format(*headers))
        return results

    def networkError(
            self, train_data, train_labels, index, results, output_type,
            val_data=None, val_labels=None, nondecr=0, min_error=None, epsilon=None, error_metric="MSE"
    ):
        y_pred = self.feedForward(train_data)  # Feed forward results to be used in training data metrics
        train_mse = MSE(y_true=train_labels, y_pred=y_pred)  # Training MSE
        results['train_mse'].append(train_mse)
        errors = [train_mse]
        if output_type != "Regression":
            train_error = self.calcError(y_pred, train_labels)  # Cross Entropy error based on optimization function
            results['train_error'].append(train_error)
            errors += [train_error]
        if output_type == 'Classification':  # Adds training data metrics for classification problems
            train_f1 = testF1(y_pred, train_labels)  # Training F1 score
            train_acc = testAccuracy(y_pred, train_labels)  # Training accuracy
            results['train_f1'].append(train_f1)
            results['train_acc'].append(train_acc)
            errors += [train_f1, train_acc]
        if val_data is not None and val_labels is not None:
            y_pred = self.feedForward(val_data)  # Feed forward results to be used in validation data metrics
            val_mse = MSE(y_true=val_labels, y_pred=y_pred)  # Validation MSE
            results['val_mse'].append(val_mse)
            errors += [val_mse]
            if output_type != "Regression":
                val_error = self.calcError(y_pred, val_labels)  # Cross Entropy error based on optimization function
                results['val_error'].append(val_error)
                errors += [val_error]
            if output_type == 'Classification':  # Adds validation data metrics for classification
                val_f1 = testF1(y_pred, val_labels)  # Validation F1 score
                val_acc = testAccuracy(y_pred, val_labels)  # Validation Accuracy
                results['val_f1'].append(val_f1)
                results['val_acc'].append(val_acc)
                errors += [val_f1, val_acc]
            stop_error = val_mse if error_metric == 'MSE' else val_error
            condition = stop_error >= min_error or np.abs(stop_error - min_error) < epsilon
            nondecr = nondecr + 1 if condition else 0
            self.best_weights = self.convertToJSON() if stop_error < min_error else self.best_weights
            min_error = min(min_error, stop_error)
        r_format = "{:<8}" + "{:<23}" * (len(errors))
        print(r_format.format(index, *errors))
        return nondecr, min_error

    def feedForward(self, entry_data):
        res = entry_data
        for key in self.layers.keys():
            res = np.dot(self.layers[key].weights, res)
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
        delta[f"dW{n_lay}"] = 1 / m * np.dot(dZ, self.layers[f"L{n_lay - 1}"].activations.T)
        delta[f"db{n_lay}"] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        delta[f"dZ{n_lay}"] = dZ
        n_lay -= 1

        # Delta propagation
        while n_lay >= 0:
            term1 = 1 / m * np.dot(self.layers[f"L{n_lay + 1}"].weights.T, delta[f"dZ{n_lay + 1}"])
            term2 = np.multiply(self.layers[f"L{n_lay}"].activations, (1 - self.layers[f"L{n_lay}"].activations))
            dZ = np.multiply(term1, term2)
            if n_lay != 0:
                activations = self.layers[f"L{n_lay - 1}"].activations
            else:
                activations = entry_data
            delta[f"dW{n_lay}"] = 1 / m * np.dot(dZ, activations.T)
            delta[f"db{n_lay}"] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            delta[f"dZ{n_lay}"] = dZ
            n_lay -= 1

        # Weight update
        for i, key in enumerate(self.layers.keys()):
            self.layers[key].weights = self.layers[key].weights - alpha * delta[f"dW{i}"]
            self.layers[key].bias = self.layers[key].bias - alpha * delta[f"db{i}"]

    def calcError(self, y_pred, y_true):
        m = y_pred.shape[1]
        error = -1 / m * (np.multiply(y_true, np.log2(y_pred)) + np.multiply(1 - y_true, np.log2(1 - y_pred)))
        return np.sum(np.sum(error))

    def printNetwork(self):
        for i, key in enumerate(self.layers):
            print(f"Layer {i}", end='')
            print("-" * 100)
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

    def saveAsJSON(self, path=None):
        if path is None:
            path = f'./Networks/Network-{self.name}.json'
        network = self.convertToJSON()
        with open(path, 'w') as file:
            json.dump(network, file, indent=4)

    def convertToJSON(self):
        network = {"entradas": len(self.layers['L0'].weights[0]), "capas": []}
        for layer in self.layers:
            network['capas'].append({
                "neuronas": [
                    {"pesos":
                         list(self.layers[layer].bias[i]) + list(weight)
                     } for i, weight in enumerate(self.layers[layer].weights)
                ]
            })
        return network

    def saveAsExcel(self, path):
        if path is None:
            path = f'./Networks/{self.name}.xlsx'
        writer = pd.ExcelWriter(path, engine='openpyxl')
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
