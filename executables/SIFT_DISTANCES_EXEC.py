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
n_neighbors = len(descriptors)
knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean')

descriptors_train = descriptors[:, 2:]
knn.fit(descriptors_train)

counts = {}

new_image_path = list(images_path.values())[random.randrange(0, len(carpetas)*n_imgs - 1)]
new_img = cv2.cvtColor(cv2.imread(new_image_path), cv2.COLOR_BGR2RGB)
_ , descriptor = sift.detectAndCompute(new_img, mask=None)
descriptors_arr = np.array(descriptor)
for descriptor_i in descriptor:
    descriptor_i = descriptor_i.reshape(1, -1)
    distance, indice = knn.kneighbors(descriptor_i, n_neighbors=n_neighbors)
    idx_dist = list(zip(indice[0], distance[0]))
    for tupl in idx_dist:
        carpeta = str(int(descriptors[tupl[0],0]))
        num_img = str(int(descriptors[tupl[0],1]))
        id = (carpeta, num_img)
        if id in counts:
            counts[id] += [tupl[1]]
        else:
            counts[id] = [tupl[1]]
            
            
avg_distances = {clave: mean(valores) for clave, valores in counts.items()}
sorted_indexes = get_n_similar(7, avg_distances, 0)