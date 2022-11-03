import tensorflow as tf
import numpy as np
import matplotlib.pyplot as mp
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# suprimir warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# dataset
dataset_diretorio = "C:\\Users\\henri\\Downloads\\RN_ReconhecimentoClubes\\clubes"

# redimensionamento
batch_size = 32
img_height = 180
img_width = 180

# dataset treino
treino_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_diretorio,
    validation_split=0.2,
    subset="training",
    seed=123,  # valor para embaralhamento dos dados, necessário devido à fatia de validação
    image_size=(img_height, img_width),
    batch_size=batch_size)

# dataset validação
validacao_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_diretorio,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

nome_classes = treino_dataset.class_names

AUTOTUNE = tf.data.AUTOTUNE

treino_dataset = treino_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validacao_dataset = validacao_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Aplicar aumento de dados
aumento_dados = keras.Sequential(
    [
        layers.RandomFlip("horizontal",  # virar imagem randomicamente
                          input_shape=(img_height,
                                       img_width,
                                       3)),
        layers.RandomRotation(0.1),  # girar imagem randomicamente
        layers.RandomZoom(0.1),  # ampliar imagem randomicamente
    ]
)

# modelo
qtd_classes = len(nome_classes)

modelo = Sequential([
    aumento_dados,
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),  # normalizar dados
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.1),  # aplicar dropout
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(qtd_classes)
])

modelo.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])  # imprimir precisão do treinamento e validação em épocas

modelo.summary()  # imprimir informações do modelo

# treinar modelo
epocas = 15
historico_modelo = modelo.fit(
    treino_dataset,
    validation_data=validacao_dataset,
    epochs=epocas
)

# verificar se está havendo overfitting ou underfitting
precisao_treino = historico_modelo.history['accuracy']
precisao_validacao = historico_modelo.history['val_accuracy']

perda_treino = historico_modelo.history['loss']
perda_validacao = historico_modelo.history['val_loss']

range_epocas = range(epocas)

mp.figure(figsize=(8, 8))
mp.subplot(1, 2, 1)
mp.plot(range_epocas, precisao_treino, label='Precisão Treino')
mp.plot(range_epocas, precisao_validacao, label='Precisão Validação')
mp.legend(loc='lower right')
mp.title('Precisão Treino e Validação')

mp.subplot(1, 2, 2)
mp.plot(range_epocas, perda_treino, label='Treino Perda')
mp.plot(range_epocas, perda_validacao, label='Validação Perda')
mp.legend(loc='upper right')
mp.title('Perda Treino e Validação')
mp.show()

# classificar
while True:
    testar = input("\nTestar imagem?(s/n)")
    if testar == 'n':
        break
    # pedir caminho da imagem
    Tk().withdraw()
    imagem_dir = askopenfilename()

    img = tf.keras.utils.load_img(
        imagem_dir, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predicoes = modelo.predict(img_array)
    pontuacao = tf.nn.softmax(predicoes[0])

    print(
        "\nImagem classificada como: {}.\nPrecisão: {:.2f}%."
        .format(nome_classes[np.argmax(pontuacao)], 100 * np.max(pontuacao))
    )
