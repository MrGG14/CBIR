import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
from sklearn.neighbors import NearestNeighbors
def get_n_similar(n, counts, reverse):
    sorted_counts = sorted(counts.items(), key=lambda x:x[1], reverse=reverse)[:n] #Ordeno y me quedo con los 5 más parecidos
    sorted_indexes = [idx[0] for idx in sorted_counts]
    return sorted_indexes

descriptors = np.load('./npy_mat/SIFT_descriptors.npy')

sift = cv2.SIFT_create()
n_neighbors = 50

knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean')

descriptors_train = descriptors[:, 2:] #Quitamos los dos primeros pertenecientes a indices de la imagen
knn.fit(descriptors_train)

counts = {}
#LEER IMAGEN DE CONSULTA
new_image_path = list(images_path.values())[random.randrange(0, len(carpetas)*n_imgs - 1)]
new_img = cv2.cvtColor(cv2.imread(new_image_path), cv2.COLOR_BGR2RGB)

_ , new_descriptors = sift.detectAndCompute(new_img, mask=None)
for descriptor in new_descriptors:
    descriptor = descriptor.reshape(1, -1)
    distance, indice = knn.kneighbors(descriptor, n_neighbors=n_neighbors)
    for idx in indice[0]:
        carpeta = str(int(descriptors[idx,0]))
        num_img = str(int(descriptors[idx,1]))
        id = (carpeta, num_img)
        if id in counts:
            counts[id] += 1
        else:
            counts[id] = 1
            
idx_similar = get_n_similar(5, counts, 1)