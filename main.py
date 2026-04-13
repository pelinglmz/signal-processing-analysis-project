import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import mahotas

DATASET_PATH = "dataset"
IMG_SIZE = 256
OUTPUT_ARFF = "glcm_12.arff"

features = []
labels = []

class_names = sorted(os.listdir(DATASET_PATH))
class_names = [c for c in class_names if os.path.isdir(os.path.join(DATASET_PATH, c))]
print("Sınıflar:", class_names)

def extract_glcm_features(img):
    img = (img / 4).astype(np.uint8)

    glcm = graycomatrix(
        img,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=64,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    asm = graycoprops(glcm, 'ASM').mean()

    haralick = mahotas.features.haralick(img, return_mean=True)
    entropy = haralick[1]
    idm = haralick[4]
    cluster_shade = haralick[7]
    cluster_prominence = haralick[8]
    diff_var = haralick[10]
    max_prob = np.max(glcm)

    return [contrast, dissimilarity, homogeneity, energy, correlation, asm,
            entropy, idm, cluster_shade, cluster_prominence, diff_var, max_prob]

for class_name in class_names:
    class_path = os.path.join(DATASET_PATH, class_name)
    for file in os.listdir(class_path):
        img_path = os.path.join(class_path, file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

        feat = extract_glcm_features(gray)
        features.append(feat)
        labels.append(class_name)

# ARFF YAZMA
with open(OUTPUT_ARFF, "w") as f:
    f.write("@RELATION texture_glcm\n\n")

    attr_names = [
        "contrast","dissimilarity","homogeneity","energy","correlation","ASM",
        "entropy","IDM","cluster_shade","cluster_prominence","diff_variance","max_probability"
    ]

    for a in attr_names:
        f.write(f"@ATTRIBUTE {a} NUMERIC\n")

    class_str = ",".join(class_names)
    f.write(f"@ATTRIBUTE class {{{class_str}}}\n\n")
    f.write("@DATA\n")

    for feat, lab in zip(features, labels):
        line = ",".join([str(x) for x in feat]) + "," + lab
        f.write(line + "\n")

print("ARFF dosyası oluşturuldu:", OUTPUT_ARFF)
print("Toplam örnek:", len(features))
print("Özellik sayısı:", len(features[0]))

