import pandas as pd
from ArtificialNeuralNetwork import ArtificialNeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import sys


def main():
    data_path = './Datasets/part2_train_data.csv'
    if len(sys.argv) > 1:
        data_path = sys.argv[1]

    data = pd.read_csv(data_path)
    X = data.loc[:, [col for col in data.columns if col not in ['y1', 'y2']]].to_numpy().T
    labels = data.loc[:, ['y1', 'y2']].to_numpy().T
    dimensions = [len(X), 2, 2]
    res_matrix = []
    writer = pd.ExcelWriter('./Networks/Parte2/errors.xlsx', engine='openpyxl')
    for i in range(20):
        print(f"--------Network {i+1}-----------------------------------------------------------------------------")
        ann = ArtificialNeuralNetwork(dimensions, name=f"Parte2-Network{i+1}")
        results = ann.trainNetwork(X, labels, max_epochs=100, output_type="Other")
        res_df = pd.DataFrame(results)
        res_df.to_excel(excel_writer=writer, sheet_name=f'Network{i+1}')
        res_matrix.append(np.array(results['train_mse']))
        ann.printNetwork()
        ann.saveAsJSON(path=f'./Networks/Parte2/JSON/Network-{ann.name}.json')
        ann.saveAsExcel(path=f'./Networks/Parte2/Excel/{ann.name}.xlsx')

    writer.save()
    writer.close()
    print("Training and validation errors saved in './Networks/Parte2/errors.xlsx'")

    lowest_mse = np.amin(res_matrix, axis=0)
    highest_mse = np.amax(res_matrix, axis=0)
    avg_mse = np.sum(res_matrix, axis=0) / len(res_matrix)

    plt.figure(figsize=(10, 7))
    plt.title("MSE per Epoch - Neural Network", fontsize=24)
    plt.plot(lowest_mse, label="Lowest MSE p/ Epoch", lw=3, ls=":")
    plt.plot(highest_mse, label="Highest MSE p/ Epoch", lw=3, ls="--")
    plt.plot(avg_mse, label="Average MSE p/ Epoch", lw=3)
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("MSE", fontsize=20)
    plt.legend(fontsize=13)
    plt.show()


if __name__ == "__main__":
    main()