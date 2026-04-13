import os, cv2, numpy as np
from skimage.feature import graycomatrix, graycoprops
import mahotas
import pywt

DATASET_PATH = "dataset"
IMG_SIZE = 256
OUTPUT_ARFF = "wavelet_glcm.arff"

features, labels = [], []
classes = [c for c in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH,c))]

def extract_features(img):
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    sub = (LL/4).astype(np.uint8)
    sub = np.clip(sub, 0, 63)


    glcm = graycomatrix(sub, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 64, True, True)

    contrast = graycoprops(glcm,'contrast').mean()
    dissim = graycoprops(glcm,'dissimilarity').mean()
    homo = graycoprops(glcm,'homogeneity').mean()
    energy = graycoprops(glcm,'energy').mean()
    corr = graycoprops(glcm,'correlation').mean()
    asm = graycoprops(glcm,'ASM').mean()

    har = mahotas.features.haralick(sub, return_mean=True)
    return [contrast,dissim,homo,energy,corr,asm,har[1],har[4],har[7],har[8],har[10],np.max(glcm)]

for cl in classes:
    for f in os.listdir(os.path.join(DATASET_PATH,cl)):
        img = cv2.imread(os.path.join(DATASET_PATH,cl,f))
        if img is None: continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g,(IMG_SIZE,IMG_SIZE))
        features.append(extract_features(g))
        labels.append(cl)

with open(OUTPUT_ARFF,"w") as f:
    f.write("@RELATION wavelet_glcm\n\n")
    for n in ["contrast","dissim","homo","energy","corr","asm","entropy","idm","shade","prom","diffvar","maxprob"]:
        f.write(f"@ATTRIBUTE {n} NUMERIC\n")
    f.write(f"@ATTRIBUTE class {{{','.join(classes)}}}\n\n@DATA\n")
    for ft,lb in zip(features,labels):
        f.write(",".join(map(str,ft))+","+lb+"\n")

print("Wavelet ARFF oluşturuldu.")
