from ArtificialNeuralNetwork import ArtificialNeuralNetwork
import numpy as np

def main():
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Resultados usando pesos aleatorios
    ann = ArtificialNeuralNetwork(dimensions=[2, 2, 2], name="Parte1-FeedForward")
    ann.saveAsJSON(path=f'./Networks/Parte1/JSON/Network-{ann.name}.json')
    ann.saveAsExcel(path=f'./Networks/Parte1/Excel/{ann.name}.xlsx')
    ann.printNetwork()

    # Resultados usando los pesos provistos en “part1_red_prueba.json”
    json_ann = ArtificialNeuralNetwork(jsonPath='./Datasets/part1_red_prueba.json')
    json_ann.printNetwork()


if __name__ == "__main__":
    main()