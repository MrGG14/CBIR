def calculate_texture_histogram(image, bins=8):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histograms = []

    for i in range(bins):
        lower_bound = i * (256 // bins)
        upper_bound = (i + 1) * (256 // bins)
        mask = cv2.inRange(gray_image, lower_bound, upper_bound)
        hist = cv2.calcHist([gray_image], [0], mask, [8], [0, 256])
        histograms.append(hist)

    histogram = np.concatenate(histograms)
    histogram = cv2.normalize(histogram, None).flatten()

    return histogram

# Función para calcular los descriptores de esquinas Harris de una imagen
def calculate_harris_descriptors(image, max_corners=100):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectar esquinas Harris
    dst = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)

    # Normalizar la matriz de esquinas
    norm_dst = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX)

    # Encontrar las coordenadas de las esquinas
    corners = np.argwhere(norm_dst > 150)  # Puedes ajustar este umbral según tus necesidades

    # Tomar un máximo de 'max_corners' esquinas
    if len(corners) > max_corners:
        np.random.shuffle(corners)
        corners = corners[:max_corners]

    return corners




def get_n_similar(n, counts, reverse):
    sorted_counts = sorted(counts.items(), key=lambda x:x[1], reverse=reverse)[:n] #Ordeno y me quedo con los 5 más parecidos
    sorted_indexes = [idx[0] for idx in sorted_counts]
    return sorted_indexes

def display_n_similar(sorted_indexes, n):
    if n <= 0:
        print("El valor de n debe ser mayor que 0.")
        return

    if n > len(sorted_indexes):
        print(f"Solo hay {len(sorted_indexes)} imágenes en la lista. Mostrando todas.")
        n = len(sorted_indexes)
    fig, axs = plt.subplots(1, n, figsize=(15,15))
    for i in range(n):
        n_carpeta = (sorted_indexes[i][0])
        n_img = (sorted_indexes[i][1])
        path = images_path[(n_carpeta, n_img)]
        im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        axs[i].imshow(im)
        axs[i].axis('off')
        titulo = f'{n_carpeta}_{n_img}'
        axs[i].set_title(titulo)
        

    plt.show()
    
def get_images_path(carpetas, n_imgs):
    images_path = {}
    for carpeta in carpetas:
        for i in range(n_imgs):
            images_path[(carpeta[-7:], str(i))] = f'./dataset/{carpeta}/{carpeta[-9:]}_{str(i)}.JPEG'
    return images_path #Devuelve un diccionario en el que la clave es (carpeta, id), y los valores los paths