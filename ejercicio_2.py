#Ejercicio 2:

import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('lenna2.png', cv2.IMREAD_GRAYSCALE)
if img1 is None or img2 is None:
    print ('Hubo un error al cargar las imagenes')
else:
    print('Las imagenes se cargaron exitosamente')
orb = cv2.ORB_create()

p_clave1, descriptores1 = orb.detectAndCompute(img1, None)
p_clave2, descriptores2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
emparejamientos = bf.match(descriptores1, descriptores2)
emparejamientos = sorted(emparejamientos, key=lambda x:x.distance)
img_emparejada = cv2.drawMatches(img1, p_clave1, img2, p_clave2, emparejamientos[:10], None, flags=2)

plt.f(figsize=(12, 6))
plt.imshow(img_emparejada)
plt.show()