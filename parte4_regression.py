import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt
from ArtificialNeuralNetwork import ArtificialNeuralNetwork


def main():
    training_data = pd.read_csv('./Datasets/part4_pokemon_go_train.csv')
    training_data.drop(['nombre'], axis=1, inplace=True)
    preX_train = training_data.loc[:, training_data.columns != 'PC']
    train_label = training_data['PC'].to_numpy().T
    train_label = np.reshape(train_label, (1, len(train_label)))
    val_data = pd.read_csv('./Datasets/part4_pokemon_go_validation.csv')
    val_data.drop(['nombre'], axis=1, inplace=True)
    preX_val = val_data.loc[:, val_data.columns != 'PC']
    val_label = val_data['PC'].to_numpy().T
    val_label = np.reshape(val_label, (1, len(val_label)))
    testing_data = pd.read_csv('./Datasets/part4_pokemon_go_test.csv')
    testing_data.drop(['nombre'], axis=1, inplace=True)
    preX_test = testing_data.loc[:, testing_data.columns != 'PC']
    test_label = testing_data['PC'].to_numpy().T
    test_label = np.reshape(test_label, (1, len(test_label)))

    X_train = preprocess(preX_train).to_numpy().T
    X_val = preprocess(preX_val).to_numpy().T
    X_test = preprocess(preX_test).to_numpy().T

    entries_number = X_train.shape[0]
    networks = [
        [entries_number, 16, 1],
        [entries_number, 32, 1],
        [entries_number, 16, 16, 1],
        [entries_number, 32, 32, 1],
        [entries_number, 64, 1],
        [entries_number, 128, 1],
        [entries_number, 256, 1]
    ]
    learning_rate = [0.00003, 0.00003, 0.000001, 0.000001, 0.0001, 0.0001, 0.0001, 0.0001]
    netnames = ['16', '32', '16-16', '32-32', '64', '128', '256', '512']
    res_dict = {'hidden_layers': [], 'test_mse': []}
    writer = pd.ExcelWriter('./Networks/Parte4/errors.xlsx', engine='openpyxl')
    for i, network in enumerate(networks):
        print(f"--------Network {i + 1}-----------------------------------------------------------------------------")
        ann = ArtificialNeuralNetwork(network, name=f"Network{netnames[i]}", output_funct="Linear")
        results = ann.trainNetwork(
            training_data=X_train, training_labels=train_label, max_epochs=50,
            val_data=X_val, val_labels=val_label, max_nondecreasing=10, epsilon=0.001, alpha=0.0001,
            output_type="Regression", error_metric="MSE"
        )
        res_df = pd.DataFrame(results)
        res_df.to_excel(excel_writer=writer, sheet_name=f'Network{i + 1}')
        plotResults(
            results['train_mse'], results['val_mse'], title=f"MSE per Epoch - {netnames[i]}",
            x_label="Epoch", y_label="MSE"
        )
        ann.printNetwork()
        ann.saveAsJSON(path=f'./Networks/Parte4/JSON/Network-{ann.name}.json')
        ann.saveAsExcel(path=f'./Networks/Parte4/Excel/{ann.name}.xlsx')
        test_mse = testResults(X_test, test_label, ann)
        res_dict['hidden_layers'].append(netnames[i])
        res_dict['test_mse'].append(test_mse)

    res_df = pd.DataFrame(data=res_dict)
    print(res_df.head(10))
    res_df.to_excel('./Networks/Parte4/tests.xlsx')

    writer.save()
    writer.close()


def preprocess(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return pd.DataFrame(scaler.transform(data), columns=data.columns)


def testResults(test_data, test_labels, ann):
    predictions = ann.feedForward(test_data)
    mse = MSE(test_labels, predictions)
    return mse


def plotResults(train_res, val_res, title, x_label, y_label):
    plt.figure(figsize=(10, 7))
    plt.title(title, fontsize=24)
    plt.plot(train_res, label=f"Train {title}", lw=3)
    plt.plot(val_res, label=f"validation {title}", lw=3, ls='--')
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.legend(fontsize=13)
    plt.show()


if __name__ == "__main__":
    main()