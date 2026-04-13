import os, cv2, numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
import mahotas

DATASET_PATH = "dataset"
IMG_SIZE = 256

def soft_hist_features(img):
    return [img.mean(), img.std(), np.mean((img-img.mean())**3), np.mean((img-img.mean())**4)]

def lbp_features(img):
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    hist,_ = np.histogram(lbp, bins=10, range=(0,10), density=True)
    return hist.tolist()

def hog_features(img):
    h = hog(img, pixels_per_cell=(16,16), cells_per_block=(2,2), orientations=9, feature_vector=True)
    return h[:50].tolist()   # boyutu sınırlıyoruz

def glcm_features(img):
    img = (img/4).astype(np.uint8)
    glcm = graycomatrix(img,[1],[0,np.pi/4,np.pi/2,3*np.pi/4],64,True,True)
    props = ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']
    feats = [graycoprops(glcm,p).mean() for p in props]
    har = mahotas.features.haralick(img, return_mean=True)
    return feats + [har[1],har[4],har[7],har[8],har[10],np.max(glcm)]

def process(OUT):
    X,Y = [],[]
    classes = [c for c in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH,c))]
    for cl in classes:
        for f in os.listdir(os.path.join(DATASET_PATH,cl)):
            img = cv2.imread(os.path.join(DATASET_PATH,cl,f),0)
            img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            feat = glcm_features(img)
            if "lbp" in OUT: feat += lbp_features(img)
            if "hog" in OUT: feat += hog_features(img)
            if "soft" in OUT: feat += soft_hist_features(img)
            X.append(feat); Y.append(cl)
    with open(OUT,"w") as f:
        f.write("@RELATION fusion\n\n")
        for i in range(len(X[0])): f.write(f"@ATTRIBUTE f{i} NUMERIC\n")
        f.write(f"@ATTRIBUTE class {{{','.join(classes)}}}\n\n@DATA\n")
        for a,b in zip(X,Y): f.write(",".join(map(str,a))+","+b+"\n")
    print(OUT,"oluşturuldu. Özellik sayısı:",len(X[0]))

process("glcm_12.arff")
process("glcm_lbp.arff")
process("glcm_lbp_hog_soft.arff")
