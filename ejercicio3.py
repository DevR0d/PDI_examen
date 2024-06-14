#Calcular y mostrar la Transformada de Fourier de una imagen en escala de grises utilizando Python con NumPy y OpenCV. 
#Muestra el espectro de magnitud.
import cv2
import numpy as np
import matplotlib.pyplot as plt

imagen = cv2.imread('fisi.jpg')

if imagen is None:
    print("Hubo un error al cargar la imagen :( ")
else:
    print("La imagen se cargo exitosamente")
    
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    TransDF = cv2.dft(np.float32(imagen), flags = cv2.DFT_COMPLEX_OUTPUT)
    TransDF_shift = np.fft.fftshift(TransDF)
    
    E_mag = 20*np.log(cv2.magnitude(TransDF_shift[:, :, 0], TransDF_shift[:, :, 1]))
    
    plt.figure(figsize=(5,5))
    
    plt.subplot(2, 2, 1)
    plt.title('Imagen Original en escala gris')
    plt.imshow(imagen, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title('Espectro de Magnitud')
    plt.imshow(E_mag, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()