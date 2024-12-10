import argparse
from glob import glob
from os.path import join, isdir, isfile

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description='This script visualizes estimated clusters.')

parser.add_argument('--clusters_path', type=str, help='File or directory path to generated clusters.')
parser.add_argument('--src_img', type=str, help='Path to the source image.')
parser.add_argument('--dst_img', type=str, help='Path to the image to be written.')

args = parser.parse_args()


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


def rgb_to_cmyk(rgb):
    """Convert RGB array to CMYK."""
    rgb = rgb / 255.0
    K = 1 - np.max(rgb, axis=-1)
    C = (1 - rgb[..., 0] - K) / (1 - K + 1e-10)
    M = (1 - rgb[..., 1] - K) / (1 - K + 1e-10)
    Y = (1 - rgb[..., 2] - K) / (1 - K + 1e-10)
    C, M, Y, K = (C * 255, M * 255, Y * 255, K * 255)
    return np.stack([C, M, Y, K], axis=-1).astype(np.uint8)


def main(clusters_path, src_img, dst_img):
    clusters = load_clusters(clusters_path)  # Clusters đã có dạng CMYK
    img = Image.open(src_img).convert('RGB')  # Đảm bảo ảnh nguồn ở định dạng RGB
    img = np.array(img)
    shape = img.shape

    # Chuyển ảnh từ RGB sang CMYK
    img_cmyk = rgb_to_cmyk(img.reshape(-1, 3))

    # Ánh xạ từng pixel đến cụm gần nhất
    new_image = np.zeros_like(img_cmyk)
    for i in range(img_cmyk.shape[0]):
        ind = np.linalg.norm(clusters - img_cmyk[i], axis=-1).argmin()
        new_image[i] = clusters[ind].astype(np.uint8)

    # Ghi ảnh kết quả dưới dạng CMYK
    new_image = new_image.reshape((shape[0], shape[1], 4))
    cmyk_img = Image.fromarray(new_image, mode='CMYK')
    cmyk_img.save(dst_img, format='JPEG')
    
if __name__ == '__main__':
    main("/home/vu/kmeans_mapreduce-master/clusters-out.txt",
         "/home/vu/kmeans_mapreduce-master/image1.jpg",
         "/home/vu/kmeans_mapreduce-master/image2.jpg"
         )