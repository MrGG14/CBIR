import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
from sklearn.neighbors import NearestNeighbors
from statistics import mean

def get_n_similar(n, counts, reverse):
    sorted_counts = sorted(counts.items(), key=lambda x:x[1], reverse=reverse)[:n] #Ordeno y me quedo con los 5 más parecidos
    sorted_indexes = [idx[0] for idx in sorted_counts]
    return sorted_indexes

descriptors = np.load('./npy_mat/SIFT_descriptors.npy')

sift = cv2.SIFT_create()
descriptors_train = descriptors[:, 2:]

# Crear un objeto Brute-Force Matcher
bf = cv2.BFMatcher()

# Definir la imagen de consulta y extraer sus descriptores SIFT
new_img = cv2.cvtColor(cv2.imread('./dataset/test/coches-/n02814533_189.JPEG'), cv2.COLOR_BGR2RGB)
query_keypoints, query_descriptors = sift.detectAndCompute(new_img, mask=None)
descriptors_train = descriptors_train.astype(np.float32)


# Realizar la búsqueda de coincidencias entre la imagen de consulta y la base de datos
matches = bf.knnMatch(query_descriptors, descriptors_train, k=2)

# Filtrar las coincidencias con un umbral Lowe's ratio
good_matches = []
for m, n in matches:
    if m.distance < 0.85 * n.distance:
        good_matches.append(m)

pos = [m.trainIdx for m in good_matches]
ids = []
for idx in pos:
        carpeta = str(int(descriptors[idx,0]))
        num_img = str(int(descriptors[idx,1]))
        ids.append((carpeta, num_img))