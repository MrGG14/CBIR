import os
from tkinter import *
from tkinter import filedialog

import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
import numpy as np
from keras import Model

from sklearn.neighbors import NearestNeighbors
from keras.applications import VGG16
from keras.models import Model
from keras.applications.vgg16 import preprocess_input


def get_images_path(carpetas, n_imgs):
    images_path = {}
    for carpeta in carpetas:
        for i in range(n_imgs):
            images_path[(carpeta[-7:], str(i))] = f'./dataset/{carpeta}/{carpeta[-9:]}_{str(i)}.JPEG'
    return images_path  # Devuelve un diccionario en el que la clave es (carpeta, id), y los valores los paths


def calculate_color_histogram(image, bins=8):
    histograms = []
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        histograms.append(hist)
    histogram = np.concatenate(histograms)
    histogram = cv2.normalize(histogram, None).flatten()
    return histogram


def cnn():
    res_images_path = []

    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

    new_img = query_image
    target_size = (224, 224)
    new_img = cv2.resize(new_img, target_size)
    # Agregar una dimensión
    new_img = np.expand_dims(new_img, axis=0)
    # Normalizar la imagen
    new_img = preprocess_input(new_img)
    # Extraer las características de la imagen
    caracteristics_img = model.predict(new_img)
    caracteristics_flat = caracteristics_img.reshape(1, -1)

    caracteristics = np.load('./npy_mat/CNN_matrix.npy')
    caracteristics_train = caracteristics[:, 2:]

    knn = NearestNeighbors(n_neighbors=n_images.get(), algorithm='auto', metric='euclidean')
    knn.fit(caracteristics_train)

    distance, indice = knn.kneighbors(caracteristics_flat, n_neighbors=n_images.get())
    for idx in indice[0]:
        carpeta = str(int(caracteristics[idx, 0]))
        num_img = str(int(caracteristics[idx, 1]))
        res_images_path.append(images_path[(carpeta, num_img)])
    return res_images_path


def sift():
    res_images_path = []

    descriptors = np.load('./npy_mat/SIFT_descriptors.npy')
    descriptors_train = descriptors[:, 2:]  # Quitamos los dos primeros pertenecientes a indices de la imagen

    n_neighbors = 50
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean')
    knn.fit(descriptors_train)

    new_img = query_image
    sift_model = cv2.SIFT_create()
    _, new_descriptors = sift_model.detectAndCompute(new_img, mask=None)

    counts = {}
    for descriptor in new_descriptors:
        descriptor = descriptor.reshape(1, -1)
        distance, indice = knn.kneighbors(descriptor, n_neighbors=n_neighbors)
        for idx in indice[0]:
            carpeta = str(int(descriptors[idx, 0]))
            num_img = str(int(descriptors[idx, 1]))
            id = (carpeta, num_img)
            if id in counts:
                counts[id] += 1
            else:
                counts[id] = 1
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n_images.get()]
    sorted_indexes = [idx[0] for idx in sorted_counts]
    for index in sorted_indexes:
        carpeta = index[0]
        num_img = index[1]
        res_images_path.append(images_path[(carpeta, num_img)])
    return res_images_path


def color_histogram():
    res_images_path = []

    color_histograms = np.load('./npy_mat/color_histograms.npy')
    color_histograms_train = color_histograms[:, 2:]  # Quitamos los dos primeros pertenecientes a indices de la imagen

    knn = NearestNeighbors(n_neighbors=n_images.get(), algorithm='auto', metric='euclidean')
    knn.fit(color_histograms_train)

    new_img = query_image
    new_histogram = calculate_color_histogram(new_img)
    histogram_flat = new_histogram.reshape(1, -1)

    distance, indice = knn.kneighbors(histogram_flat, n_neighbors=n_images.get())
    idx_dist = list(zip(indice[0], distance[0]))
    idx_dist = [tupla[0] for tupla in sorted(idx_dist, key=lambda x: x[1])]
    for idx in idx_dist:
        carpeta = str(int(color_histograms[idx, 0]))
        num_img = str(int(color_histograms[idx, 1]))
        res_images_path.append(images_path[(carpeta, num_img)])

    return res_images_path


def harris():
    res_images_path = []

    corners = np.load('./npy_mat/HARRIS_descriptors.npy')
    descriptors_train = corners[:, 2:]  # Quitamos los dos primeros pertenecientes a indices de la imagen

    n_neighbors = 1000
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean')
    knn.fit(descriptors_train)

    new_img = query_image
    gray_im_new = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    new_corners = cv2.cornerHarris(gray_im_new, 2, 3, 0.04)

    counts = {}
    for descriptor in new_corners:
        descriptor = descriptor.reshape(1, -1)
        mn = descriptor.mean()
        std = descriptor.std()
        descriptor = (descriptor - mn) / std
        distance, indice = knn.kneighbors(descriptor, n_neighbors=100)
        for idx in indice[0]:
            carpeta = str(int(corners[idx, 0]))
            num_img = str(int(corners[idx, 1]))
            id = (carpeta, num_img)
            if id in counts:
                counts[id] += 1
            else:
                counts[id] = 1
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n_images.get()]
    sorted_indexes = [idx[0] for idx in sorted_counts]
    for index in sorted_indexes:
        carpeta = index[0]
        num_img = index[1]
        res_images_path.append(images_path[(carpeta, num_img)])

    return res_images_path


def orb():
    res_images_path = []

    corners = np.load('./npy_mat/ORB_descriptors.npy')
    descriptors_train = corners[:, 2:]  # Quitamos los dos primeros pertenecientes a indices de la imagen

    n_neighbors = 1000
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='hamming')
    knn.fit(descriptors_train)

    new_img = query_image
    gray_im_new = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    orb_model = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=10)
    keypoints = orb_model.detect(gray_im_new, None)
    # Extrae los descriptores de los keypoints
    _, descriptors = orb_model.compute(gray_im_new, keypoints)

    counts = {}
    for descriptor in descriptors:
        descriptor = descriptor.reshape(1, -1)
        distance, indice = knn.kneighbors(descriptor, n_neighbors=100)
        for idx in indice[0]:
            carpeta = str(int(corners[idx, 0]))
            num_img = str(int(corners[idx, 1]))
            id = (carpeta, num_img)
            if id in counts:
                counts[id] += 1
            else:
                counts[id] = 1
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n_images.get()]
    sorted_indexes = [idx[0] for idx in sorted_counts]
    for index in sorted_indexes:
        carpeta = index[0]
        num_img = index[1]
        res_images_path.append(images_path[(carpeta, num_img)])

    return res_images_path


def calcular_imagenes():
    # Borro las imagenes de las etiquetas en caso de que tengan
    for label in lbl_output_images:
        label.image = ""

    res_images_path = []
    if seleccion_algoritmo.get() == 1:
        res_images_path = cnn()

    elif seleccion_algoritmo.get() == 2:
        res_images_path = sift()

    elif seleccion_algoritmo.get() == 3:
        res_images_path = color_histogram()

    elif seleccion_algoritmo.get() == 4:
        res_images_path = harris()

    elif seleccion_algoritmo.get() == 5:
        res_images_path = orb()

    for i in range(len(res_images_path)):
        image = cv2.cvtColor(cv2.imread(res_images_path[i]), cv2.COLOR_BGR2RGB)
        image = imutils.resize(image, width=180)
        im = Image.fromarray(image)
        img = ImageTk.PhotoImage(image=im)

        lbl_output_images[i].configure(image=img)
        lbl_output_images[i].image = img

    # Label IMAGEN DE SALIDA
    lbl_info3 = Label(root, text="IMAGENES DE SALIDA:", font="bold")
    lbl_info3.grid(column=1, row=0, padx=5, pady=5, columnspan=6)
    pass


def elegir_imagen():
    # Especificar los tipos de archivos, para elegir solo a las imágenes
    path_image = filedialog.askopenfilename(filetypes=[
        ("image", ".jpeg"),
        ("image", ".png"),
        ("image", ".jpg")])
    if len(path_image) > 0:
        # Leer la imagen de entrada y la redimensionamos
        global query_image
        query_image = cv2.cvtColor(cv2.imread(path_image), cv2.COLOR_BGR2RGB)
        im = imutils.resize(query_image, width=300)
        im = Image.fromarray(im)
        img = ImageTk.PhotoImage(image=im)
        lbl_input_image.configure(image=img)
        lbl_input_image.image = img
        # Label IMAGEN DE ENTRADA
        lbl_info1 = Label(root, text="IMAGEN DE ENTRADA:")
        lbl_info1.grid(column=0, row=1, padx=5, pady=5)

        seleccion_algoritmo.set(0)


# Creamos la ventana
root = Tk()

n_imgs = 100
carpetas = ["autobus-n04487081", "clavos-n03804744", "coche-n02814533", "collarin-n03814639", "desatascador-n03970156",
            "gatos-n02123394", "mono-n02480495", "puentes-n04532670", "silla-n04099969", "perro-n02099601",
            "pato-n01855672", "pizza-n07873807", "mar-n09428293", "ipod-n03584254", "platano-n07753592",
            "mascara_gas-n03424325",
            "pajarita-n02883205", "mosca-n02190166", "helado-n07615774", "canon-n02950826"]
global images_path
images_path = get_images_path(carpetas, n_imgs)

global query_image

btn_elegir_imagen = Button(root, text="Elegir imagen", width=25, command=elegir_imagen)
btn_elegir_imagen.grid(column=0, row=0, padx=5, pady=5)

# Label donde se presentará la imagen de entrada
lbl_input_image = Label(root)
lbl_input_image.grid(column=0, row=2, rowspan=2)

lbl_info2 = Label(root, text="¿Qué algoritmo te gustaría utilizar?", width=25)
lbl_info2.grid(column=0, row=4, padx=5, pady=5)

# Creamos los radio buttons y la ubicación que estos ocuparán
seleccion_algoritmo = IntVar()
rad1 = Radiobutton(root, text='CNN', width=25, value=1, variable=seleccion_algoritmo)
rad2 = Radiobutton(root, text='SIFT', width=25, value=2, variable=seleccion_algoritmo)
rad3 = Radiobutton(root, text='COLOR HISTOGRAM', width=25, value=3, variable=seleccion_algoritmo)
rad4 = Radiobutton(root, text='HARRIS', width=25, value=4, variable=seleccion_algoritmo)
rad5 = Radiobutton(root, text='ORB', width=25, value=5, variable=seleccion_algoritmo)

rad1.grid(column=0, row=5)
rad2.grid(column=0, row=6)
rad3.grid(column=0, row=7)
rad4.grid(column=0, row=8)
rad5.grid(column=0, row=9)

lbl_info2 = Label(root, text="¿Cuántas imágenes quieres?", width=25)
lbl_info2.grid(column=0, row=10, padx=5, pady=5)

n_images = Scale(root, from_=1, to=12, orient=HORIZONTAL)
n_images.grid(column=0, row=11, padx=5, pady=5)

btn_calcular_imagenes = Button(root, text="Calcular imagenes", width=25, command=calcular_imagenes)
btn_calcular_imagenes.grid(column=0, row=12, padx=5, pady=5)

lbl_output_images = [Label(root), Label(root), Label(root), Label(root), Label(root), Label(root), Label(root),
                     Label(root), Label(root), Label(root), Label(root), Label(root)]

for i in range(len(lbl_output_images)):
    if i < 6:
        lbl_output_images[i].grid(column=i + 1, row=2)
    else:
        lbl_output_images[i].grid(column=i - 5, row=3)

root.mainloop()
