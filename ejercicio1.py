#Implementar la ecualización de histograma en una imagen en escala de grises utilizando Python con NumPy y OpenCV.Muestra la imagen original y la ecualizada
import cv2
import numpy as np
import matplotlib.pyplot as plt

imagen = cv2.imread('familia.png')

if imagen is None:
    print('Hubo un error al cargar la imagen')
else:
    print('La imagen se cargo exitosamente')

    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    img_ecualizada = cv2.equalizeHist(imagen_gris)
    
    #hist_original = cv2.calcHist([imagen_gris], [0], None, [256], [0, 256])
    
    #hist_ecualizada = cv2.calcHist([img_ecualizada], [0], None, [256], [0, 256])
    
    plt.figure(figsize=(10,7))
    
    plt.subplot(2,2,1)
    plt.imshow(cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2BGR))
    plt.title('Imagen Original')
    plt.axis('off')
    
    plt.subplot(2,2,3)
    plt.imshow(img_ecualizada, cmap='gray')
    plt.title('Imagen Ecualizada')
    plt.axis('off')
    
    #plt.subplot(2,2,2)
    #plt.plot(hist_original, color='black')
    #plt.title('Histograma de la Imagen Original')
    #plt.xlabel('Intensidad de los píxeles')
    #plt.ylabel('Número de píxeles')
    
    #plt.subplot(2,2,4)
    #plt.plot(hist_ecualizada, color='black')
    #plt.title('Histograma de la Imagen Ecualizada')
    #plt.xlabel('Intensidad de los píxeles')
    #plt.ylabel('Número de píxeles')
    
    plt.tight_layout()
    plt.show()