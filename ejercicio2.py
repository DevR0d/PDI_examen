#Aplicar un filtro Gaussiano para suavizar una imagen utilizando Python con NumPy y OpenCV. Muestra la imagen original y la suavizada.
import cv2
import numpy as np
import matplotlib.pyplot as plt

imagen = cv2.imread('lenna.png')

if imagen is None:
    print('Hubo un error al cargar la imagen')
else:
    print('La imagen se cargo exitosamente')

    imagen_suavizada = cv2.GaussianBlur(imagen, (5, 5), 0)
    
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen_suavizada_rgb = cv2.cvtColor(imagen_suavizada, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10,7))
    
    plt.subplot(2,2,1)
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Original')
    plt.axis('off')
    
    plt.subplot(2,2,2)
    plt.imshow(imagen_suavizada_rgb)
    plt.title('Imagen suavizada')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
