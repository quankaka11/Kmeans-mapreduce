from glob import glob
from os.path import join, isdir, isfile
from skimage.feature import hog
from PIL import Image


import cv2
import numpy as np


def load_clusters(path):
    if isdir(path):
        files = glob(join(path, 'part-r-*[0-9]'))
    elif isfile(path):
        files = [path]
    else:
        raise Exception('Invalid file path.')

    centroids = [load_nparray(file)[:, 1:] for file in files]
    centroids = np.concatenate(centroids, axis=0).reshape(-1, centroids[0].shape[-1])
    return centroids


def load_nparray(file):
    data = []
    with open(file) as f:
        for line in f:
            data.append(np.array([float(num) for num in line.split(' ')]))

    return np.stack(data).astype(np.float64)

"""#---CMYK---
def bgr_to_cmyk(image):
    bgr = image.astype(np.float32) / 255.0
    K = 1 - np.max(bgr, axis=2)
    C = (1 - bgr[:, :, 2] - K) / (1 - K + 1e-8)
    M = (1 - bgr[:, :, 1] - K) / (1 - K + 1e-8)
    Y = (1 - bgr[:, :, 0] - K) / (1 - K + 1e-8)
    C[K == 1] = 0
    M[K == 1] = 0
    Y[K == 1] = 0
    CMYK = np.stack((C, M, Y, K), axis=2) * 255.0
    return CMYK.astype(np.uint8)
#######
"""


def main(clusters_path, src_img, dst_img):
    clusters = load_clusters(clusters_path)

#------------------------------------------------------------------------------------------

    #---RGB---
    img = cv2.imread(src_img)
    shape = img.shape

    img = img.reshape((-1, 3))
    ##########


    """#---CMYK---
    # img = cv2.imread(src_img)
    # img_cmyk = bgr_to_cmyk(img)
    # shape = img_cmyk.shape
    # img = img_cmyk.reshape((-1, 4))
    ###########
    """


    """#---HOG---
    img = cv2.imread(src_img, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh grayscale
    hog_features, hog_image = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                      block_norm='L2-Hys', visualize=True, feature_vector=True)
    img = hog_features.reshape((-1, hog_features.shape[-1]))
    ###########
    """
#------------------------------------------------------------------------------------------------

    new_image = np.zeros_like(img)
    for i in range(img.shape[0]):
        ind = np.linalg.norm(clusters - img[i], axis=-1).argmin()
        #----------------------------------------------------------------

        #---RGB/CMYK---
        new_image[i] = clusters[ind].astype(np.uint8)
        ###########

        """#---HOG---
        new_image[i] = clusters[ind]
        ##########
        """
        #------------------------------------------------------------------

#-------------------------------------------------------------------------

    #---RGB---
    cv2.imwrite(dst_img, new_image.reshape(shape))
    #########


    """#---CMYK---
    result_cmyk = new_image.reshape(shape).astype(np.uint8)
    image_cmyk = Image.fromarray(result_cmyk, mode="CMYK")
    image_cmyk.save(dst_img)
    ##########
    """

    """#---HOG---
    cv2.imwrite(dst_img, hog_image)
    ###########
    """
#---------------------------------------------------------------------------


if __name__ == '__main__':
    clusters_path = "/home/vu/Kmeans-mapreduce/Input/clusters1.txt"
    src_img = "/home/vu/Kmeans-mapreduce/data_prep_scripts/sample_images/image1.jpg"
    dst_img = "/home/vu/Kmeans-mapreduce/data_prep_scripts/sample_images/image3.jpg"
    main(clusters_path, src_img, dst_img)
