#Ejercicio 1: SEGMENTACION DE IMÁGENES#
#Implementar la segmentacion de una imagen utilizando el algoritmo de k-means clustering.
#Pasos:
#1.Leer una imagen y convertirla a un espacio de color adecuado (por ejemplo RGB a L*a*b*)
#2.Aplicar el algoritmo de k- means para segentar la imagen en k regiones
#3.Visualizar la imagen segmemtada y comparar los resulatados con la imagen original.

#Tips:
#- Utilizar la libreria OpenCV para leer y mostrar la imagen.
#-Utilizar la libreria scikit-learn para aplicar k-means clustering

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

imagen = cv2.imread('lenna.png')
if imagen is None:
    print ('Hubo un error al cargar la imagen')
else:
    print('La imagen se cargo exitosamente')
    
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    i_lab = cv2.cvtColor(imagen, cv2.COLOR_RGB2LAB)
    val_pixel = i_lab.reshape((-1, 3))
    val_pixel = np.float32(val_pixel)
    
    k = 5
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(val_pixel, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    i_seg = centers[labels.flatten()]
    i_seg = i_seg.reshape(i_lab.shape)
    
    i_seg_rgb = cv2.cvtColor(i_seg, cv2.COLOR_LAB2RGB)
    
    plt.figure(figsize=(10,7))
    
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Original')
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(i_seg_rgb)
    plt.title('Imagen segmentada')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()