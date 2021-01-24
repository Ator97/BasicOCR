# USAGE
# python ocr_handwriting.py --model handwriting.model --image images/umbc_address.png

from tensorflow.keras.models import load_model
import tensorflow as tf

from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2
from tensorflow import keras
from PIL import Image
import pathlib
import matplotlib.pyplot as plt

from numpy import asarray


# Recibimos arugmentos, asi se ve chido
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained handwriting recognition model")
args = vars(ap.parse_args())

# Cargamos modelo 
model = load_model(args["model"])

# Cargamos imagen, convertimos a grises y aplicamos un blur para reducir ruido.
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Detectamos los bordes de la imagen, obtenemos los simbolos a analizar
edged = cv2.Canny(blurred, 30, 150)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]

# Vamos a guardar los simbolos aqui
chars = []


for c in cnts:
	# Obtenemos las coordenadas de los simbolos
	(x, y, w, h) = cv2.boundingRect(c)

	# Filtramos los simbolos, por su tamaño probale para descargar elementos que no son letras
	if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
		# Las rellenamos de color, blanco y el resto negro para definir las imagenes
		roi = gray[y:y + h, x:x + w]
		thresh = cv2.threshold(roi, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		(tH, tW) = thresh.shape

		# Cuadramos la imagen
		if tW > tH:
			thresh = imutils.resize(thresh, width=32)
		else:
			thresh = imutils.resize(thresh, height=32)

		# Adaptamos a 180x 180 la imagen
		(tH, tW) = thresh.shape
		dX = int(max(0, 180 - tW) / 2.0)
		dY = int(max(0, 180 - tH) / 2.0)

		# Padding para que sea de 180x180 en caso de que el paso anerior falle
		padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
			left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
			value=(0, 0, 0))
		padded = cv2.resize(padded, (180,180))

		# Adaptamlos la imagen para que tenga valores entre 1 y 0 (mas optimo para la red neuronal)
		padded = padded.astype("float32") / 255.0
		padded = np.expand_dims(padded, axis=-1)
		chars.append((padded, (x, y, w, h)))


boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype="float32")
# Donde guardaremos las imagenes
data_dir = pathlib.Path("./")

from keras.preprocessing.image import save_img

k = 0
class_names = ['0', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'D', 'E', 'F', 'G', 'H', 'L', 'N', 'P', 'Q', 'R', 'S', 'T', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'r', 't', 'u', 'v', 'w', 'x', 'y', 'z']

for i in chars:
	#In vertimos color 
	for component in i:
		for value in component:
			if value[0]*255 > 125:
				value[0] = 0
			else:
				value[0] = 255
	# Guardamos la imagen
	save_img("test"+str(k)+".png",i)
	# Le pasamos la imagen al modelo
	img = keras.preprocessing.image.load_img(
    str("test"+str(k)+".png"), target_size=(180, 180)
	)
	img_array = keras.preprocessing.image.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0) # Create a batch
	predictions = model.predict(img_array)
	score = tf.nn.softmax(predictions[0])
	# Resultados
	print(
	    "This image most likely belongs to {} with a {:.2f} percent confidence."
	    .format(class_names[np.argmax(score)], 100 * np.max(score))
	)
	k += 1
