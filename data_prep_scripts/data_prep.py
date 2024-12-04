from os.path import join
from pathlib import Path
from skimage.feature import hog
from tqdm import tqdm

import cv2
import numpy as np

def nparray_to_str(X):
    to_save = '\n'.join([' '.join(str(X[i])[1:-1].split()) for i in range(len(X))])
    return to_save

#---CMYK---
# def bgr_to_cmyk(image):
#     bgr = image.astype(np.float32) / 255.0
#     K = 1 - np.max(bgr, axis=2)
#     C = (1 - bgr[:, :, 2] - K) / (1 - K + 1e-8)
#     M = (1 - bgr[:, :, 1] - K) / (1 - K + 1e-8)
#     Y = (1 - bgr[:, :, 0] - K) / (1 - K + 1e-8)
#     C[K == 1] = 0
#     M[K == 1] = 0
#     Y[K == 1] = 0
#     CMYK = np.stack((C, M, Y, K), axis=2) * 255.0
#     return CMYK.astype(np.uint8)
##########

def kmeans_plusplus_init(points, k):

    centroids = []
    centroids.append(points[np.random.choice(range(len(points)))])

    for _ in tqdm(range(1, k), desc="Initializing centroids"):

    #----------------------------------------------------------------------------

        #---RGB/CMYK---
        distances = np.min(
            [np.sum((points - centroid) ** 2, axis=1) for centroid in centroids],
            axis=0
        )
        ##########
    #---------------------------------------------------------------------------

        # Xác suất tỷ lệ với khoảng cách
        probabilities = distances / distances.sum()
        cumulative_probs = np.cumsum(probabilities)
        r = np.random.rand()

        # Chọn tâm cụm mới dựa trên xác suất
        for j, p in enumerate(cumulative_probs):
            if r < p:
                centroids.append(points[j])
                break

    return np.array(centroids)


def main(src_img, dst_folder, k):
    # files to be created
    points_path = join(dst_folder, 'points1.txt')
    clusters_path = join(dst_folder, 'clusters1.txt')

    # create directory
    Path(dst_folder).mkdir(parents=True, exist_ok=True)

#------------------------------------------------------------------------------------------
    #---RGB---
    img = cv2.imread(src_img).reshape((-1, 3)).astype(np.float32)
    #######

    #---CMYK---
    # img = cv2.imread(src_img)
    # img_cmyk = bgr_to_cmyk(img)
    # img = img_cmyk.reshape((-1, 4)).astype(np.float32)
    #########
#------------------------------------------------------------------------------------------

    with open(points_path, 'w') as f:
        f.write(nparray_to_str(img))
    print(f'Points saved in: {points_path}')

    # generate and save uniformly sampled centroids

#------------------------------------------------------------------

    #---RGB/CMYK---
    s = kmeans_plusplus_init(img, k)

#-------------------------------------------------------------------

    tmp_labels = np.arange(1, k + 1).reshape((k, 1))
    clusters = np.hstack((tmp_labels, s))

    with open(clusters_path, 'w') as f:
        f.write(nparray_to_str(clusters))
    print(f'Centroids saved in: {clusters_path}')


if __name__ == '__main__':
    main("data_prep_scripts/sample_images/image1.jpg", "Input", 100)
