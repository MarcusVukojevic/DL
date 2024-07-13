import cv2
import numpy as np
import matplotlib.pyplot as plt
from augmentations import ImageAugmentor
from PIL import Image

# Carica le immagini
augmentator = ImageAugmentor()
immagini_aug = augmentator.apply_all_augmentations('imagenet-a/n01534433/0.000030_doormat _ doormat_0.8502354.jpg')
img1 = cv2.imread('imagenet-a/n01534433/0.000030_doormat _ doormat_0.8502354.jpg', cv2.IMREAD_GRAYSCALE)  # Immagine di riferimento (oggetto da riconoscere)
img2 = immagini_aug[7]  # Assumiamo che immagini_aug[0] sia un'immagine PIL

# Controlla se img2 è un'immagine PIL e convertila in array NumPy
if isinstance(img2, Image.Image):
    img2 = np.array(img2)

# Converti img2 in scala di grigi se necessario
if len(img2.shape) == 3:
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Inizializza l'oggetto ORB
orb = cv2.ORB_create()

# Trova i keypoints e i descrittori con ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Crea il matcher BFMatcher e trova le corrispondenze
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Ordina le corrispondenze in base alla distanza
matches = sorted(matches, key=lambda x: x.distance)

# Disegna le prime 10 corrispondenze
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Mostra l'immagine risultante
plt.figure(figsize=(12, 6))
plt.imshow(img3)
plt.title('ORB Object Recognition')
plt.show()
