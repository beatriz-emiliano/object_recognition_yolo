import cv2
import numpy as np
from csv import DictWriter

# Carregando os nomes das classes
with open('data/coco.names') as f:
    labels = [line.strip() for line in f]

# Carregando o modelo YOLO
network = cv2.dnn.readNetFromDarknet('data/yolov3.cfg',
                                     'data/yolov3.weights')

# Obtendo as camadas de saída
layers_names_all = network.getLayerNames()
layers_names_output = [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]

# Parâmetros para a detecção
probability_minimum = 0.5
threshold = 0.3

# Cores aleatórias para as caixas
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# Inicializando a webcam
camera = cv2.VideoCapture(0)

# Aguardando um curto período para a webcam inicializar
cv2.waitKey(1000)

# Definindo as variáveis de altura e largura da imagem
h, w = None, None

# Criando o arquivo CSV para armazenar os resultados
with open('dados_de_captura.csv', 'w') as arquivo:
    cabecalho = ['Detectado', 'Acuracia']
    escritor_csv = DictWriter(arquivo, fieldnames=cabecalho)
    escritor_csv.writeheader()

    while True:
        # Capturando um quadro da webcam
        _, frame = camera.read()

        if w is None or h is None:
            h, w = frame.shape[:2]

        # Construindo o blob para a entrada da rede
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Configurando a entrada da rede
        network.setInput(blob)

        # Realizando a detecção
        output_from_network = network.forward(layers_names_output)

        # Preparando listas para os resultados
        bounding_boxes = []
        confidences = []
        class_numbers = []

        for result in output_from_network:
            for detected_objects in result:
                scores = detected_objects[5:]
                class_current = np.argmax(scores)
                confidence_current = scores[class_current]

                if confidence_current > probability_minimum:
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)

        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

        if len(results) > 0:
            for i in results.flatten():
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
                colour_box_current = colours[class_numbers[i]].tolist()
                cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), colour_box_current, 2)

                text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])], confidences[i])
                cv2.putText(frame, text_box_current, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

                escritor_csv.writerow({"Detectado": text_box_current.split(':')[0], "Acuracia": text_box_current.split(':')[1]})

                print(text_box_current.split(':')[0] + " - " + text_box_current.split(':')[1])

        # Exibindo o quadro resultante
        cv2.imshow("frame", frame)

        # Verificando se o usuário pressionou a tecla 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberando os recursos
camera.release()
cv2.destroyAllWindows()