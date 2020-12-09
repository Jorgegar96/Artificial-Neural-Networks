from ArtificialNeuralNetwork import ArtificialNeuralNetwork
import numpy as np

def main():
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Resultados usando pesos aleatorios
    ann = ArtificialNeuralNetwork(dimensions=[2, 2, 2], name="Parte1-FeedForward")
    ann.printNetwork()

    print(ann.predict(data.T))

    ann.saveAsJSON()
    ann.saveAsExcel()
    # Resultados usando los pesos provistos en “part1_red_prueba.json”
    json_ann = ArtificialNeuralNetwork(jsonPath='./Datasets/part1_red_prueba.json')
    json_ann.printNetwork()

    print(json_ann.predict(data.T))


if __name__ == "__main__":
    main()