from ArtificialNeuralNetwork import ArtificialNeuralNetwork
import numpy as np
import sys

def main():
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    json_path = './Datasets/part1_red_prueba.json'
    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    print("-----Randomly Initialized" + "-"*60)
    # Resultados usando pesos aleatorios
    ann = ArtificialNeuralNetwork(dimensions=[2, 2, 2], name="Parte1-FeedForward")
    ann.saveAsJSON(path=f'./Networks/Parte1/JSON/Network-{ann.name}.json')
    ann.saveAsExcel(path=f'./Networks/Parte1/Excel/{ann.name}.xlsx')
    ann.printNetwork()
    print("-----Feed Forward Results-----")
    for entries in data:
        print(f"Entry Data=> X0:{entries[0]}, X1:{entries[1]}")
        outputs = ann.feedForward(np.reshape(entries, (len(entries),1)))
        print(f"Output: Output0:{outputs[0][0]}, Output1:{outputs[1][0]}")

    print("\n-----JSON Initialized" + "-"*60)
    # Resultados usando los pesos provistos en “part1_red_prueba.json”
    json_ann = ArtificialNeuralNetwork(jsonPath=json_path)
    json_ann.printNetwork()
    print("-----Feed Forward Results-----")
    for entries in data:
        print(f"Entry Data=> X0:{entries[0]}, X1:{entries[1]}")
        outputs = json_ann.feedForward(np.reshape(entries, (len(entries),1)))
        print(f"Output: Output0:{outputs[0][0]}, Output1:{outputs[1][0]}")


if __name__ == "__main__":
    main()