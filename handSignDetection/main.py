import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)         # cv2.VideoCapture(0)  #Webcam tanımlama
detector = HandDetector(maxHands=1) #El dedektörü
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt") #Veri setinin keras_model.h5 'e dönüştürülmüş hali

offset = 20 #Kenar uzaklıgı
imgSize = 300

folder = "Data/A"
counter = 0

labels = ["A", "B", "C", "D", "E", "F", "G", "H"]  #Data dizisi

while True:

    success, img = cap.read()
    imgOutput = img.copy() #İmg kopyasını oluşturur
    hands, img = detector.findHands(img)

    if hands:

        hand = hands[0]
        x, y, w, h = hand['bbox'] #Uzunluk tanımlamaları, bbox  min ve max değerleri sınırlar

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255 #imgWhite  ölcegi
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset] #Yalnızca el kesitini kırpma

        imgCropShape = imgCrop.shape

        aspectRatio = h / w  # (hight/width)

        if aspectRatio > 1:

            k = imgSize / h
            wCal = math.ceil(k * w) # Width hesaplaması
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2) #Width gap (boşluk)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False) #Eldeki indexlere dayalı tahmin yapar
            print(prediction, "Labels İndex :", index)

        else:

            k = imgSize / w
            hCal = math.ceil(k * h) #Hight hesaplaması
            imgResize = cv2.resize(imgCrop, (imgSize, hCal)) #Yeniden boyutlandırma
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2) #hight gap (boşluk)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)


        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50), (64, 224, 208),cv2.FILLED)  # Çerçeve
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (25, 25, 25),3)  # Webcam üzerinde labels dizisinden yakalanan indexi verir.
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (64, 224, 208),4)  # Eli çevreleyen çerçeve

        cv2.imshow("ImageCrop", imgCrop) #Kırpılmış pencere
        cv2.imshow("ImageWhite", imgWhite) #Beyaz pencere

    cv2.imshow("Webcam", imgOutput) #Webcam
    #cv2.waitKey(1)

    key = cv2.waitKey(1)

    # ESC tuşuna basıldığında pencereyi kapatır
    if key == 27:
        break

#Pencereleri kapatma
hands.close()
cap.release()
cv2.destroyAllWindows()

"""if cv2.waitKey(5) & 0xFF == 27:
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
"""



