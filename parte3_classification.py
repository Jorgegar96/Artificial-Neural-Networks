import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sb
from ArtificialNeuralNetwork import ArtificialNeuralNetwork, testF1, testAccuracy, testF1PerClass, getConfusionMatrix

def main():
    training_data = pd.read_csv('./Datasets/part3_data_train.csv')
    preX_train = training_data.loc[:, training_data.columns != 'clase']
    train_label = pd.get_dummies(training_data['clase']).to_numpy().T
    val_data = pd.read_csv('./Datasets/part3_data_val.csv')
    preX_val = val_data.loc[:, val_data.columns != 'clase']
    val_label = pd.get_dummies(val_data['clase']).to_numpy().T
    testing_data = pd.read_csv('./Datasets/part3_data_test.csv')
    preX_test = testing_data.loc[:, testing_data.columns != 'clase']
    test_label = pd.get_dummies(testing_data['clase']).to_numpy().T

    le = LabelEncoder()
    le.fit(training_data['clase'].values)
    labels = le.classes_

    X_train = preprocess(preX_train).to_numpy().T
    X_val = preprocess(preX_val).to_numpy().T
    X_test = preprocess(preX_test).to_numpy().T

    entries_number = X_train.shape[0]
    networks = [
        [entries_number, 4, 5],
        [entries_number, 16, 5],
        [entries_number, 32, 5],
        [entries_number, 64, 5],
        [entries_number, 56, 5]
    ]

    res_dict = {
        'l0_neurons': [], 'test_acc': [], 'avgtest_f1': [], 'test_f1_C1': [],
        'test_f1_C2': [], 'test_f1_C3': [], 'test_f1_C4': [], 'test_f1_C5': []
    }
    writer = pd.ExcelWriter('./Networks/Parte3/errors.xlsx', engine='openpyxl')
    for i, network in enumerate(networks):
        print(f"--------Network {i+1}-----------------------------------------------------------------------------")
        netName = networks[i][1]
        ann = ArtificialNeuralNetwork(network, name=f"Network{netName}")
        results = ann.trainNetwork(
            training_data=X_train, training_labels=train_label, max_epochs=50,
            val_data=X_val, val_labels=val_label, max_nondecreasing=3, epsilon=0.001,
            output_type="Classification", error_metric="CrossEntropy"
        )
        res_df = pd.DataFrame(results)
        res_df.to_excel(excel_writer=writer, sheet_name=f'Network{i+1}')
        plotResults(
            results['train_mse'], results['val_mse'], title=f"MSE per Epoch - {netName}", x_label="Epoch", y_label="MSE"
        )
        ann.printNetwork()
        ann.saveAsJSON(path=f'./Networks/Parte3/JSON/Network-{ann.name}.json')
        ann.saveAsExcel(path=f'./Networks/Parte3/Excel/{ann.name}.xlsx')
        test_acc, test_f1, f1_per_class = testResults(X_test, test_label, ann, labels)
        res_dict['l0_neurons'].append(netName)
        res_dict['test_acc'].append(test_acc)
        res_dict['avgtest_f1'].append(test_f1)
        for j in range(5):
            res_dict[f'test_f1_C{j+1}'].append(f1_per_class[j])

    res_df = pd.DataFrame(data=res_dict)
    print(res_df.head(10))
    res_df.to_excel('./Networks/Parte3/tests.xlsx')

    writer.save()
    writer.close()


def testResults(test_data, test_labels, ann, labels):
    predictions = ann.feedForward(test_data)
    accuracy = testAccuracy(predictions, test_labels)
    f1_score = testF1(predictions, test_labels)
    f1_per_class = []
    for i in range(5):
        f1_per_class.append(testF1PerClass(predictions, test_labels, pos_class=i))
    confm = getConfusionMatrix(predictions, test_labels)
    plotConfusionMatrix(confm, labels=labels, title=f"Confusion Matrix - {ann.name}")
    return accuracy, f1_score, f1_per_class


def plotConfusionMatrix(confm, labels, title):
    plt.title(title, fontsize=18)
    heatmap = sb.heatmap(confm, annot=True, cmap='Blues', fmt='', cbar=False, xticklabels=labels, yticklabels=labels)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=30)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=30)
    plt.xlabel("True Label", fontsize=16)
    plt.ylabel("Predicted Label", fontsize=16)
    plt.show()


def plotTestResults(test_res, title, x_label, y_label):
    plt.figure(figsize=(10, 7))
    plt.title(title, fontsize=24)
    plt.plot(test_res)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.show()


def plotResults(train_res, val_res, title, x_label, y_label):
    plt.figure(figsize=(10, 7))
    plt.title(title, fontsize=24)
    plt.plot(train_res, label=f"Train {title}", lw=3)
    plt.plot(val_res, label=f"validation {title}", lw=3, ls='--')
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.legend(fontsize=13)
    plt.show()


def preprocess(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return pd.DataFrame(scaler.transform(data), columns=data.columns)


if __name__ == "__main__":
    main()
