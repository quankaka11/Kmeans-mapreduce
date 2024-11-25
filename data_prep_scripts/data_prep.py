import argparse
from os.path import join
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser(description='This script creates points.txt and clusters.txt files for a given image.')

parser.add_argument('--src_img', type=str, help='Path to the source image.')
parser.add_argument('--dst_folder', type=str, help='Directory in which points.txt and clusters.txt will be saved.')
parser.add_argument('--k_init_centroids', type=int, help='How many initial uniformly sampled centroids to generate.',
                    default=10)

args = parser.parse_args()


def nparray_to_str(X):
    return '\n'.join([' '.join(map(str, row)) for row in X])

def kmeans_plusplus_init(points, k):

    centroids = []
    # Chọn tâm cụm đầu tiên ngẫu nhiên từ dữ liệu
    centroids.append(points[np.random.choice(range(len(points)))])

    for _ in range(1, k):
        # Tính khoảng cách nhỏ nhất từ mỗi điểm đến các centroid đã chọn
        distances = np.min(
            [np.sum((points - centroid) ** 2, axis=1) for centroid in centroids],
            axis=0
        )
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
    points_path = join(dst_folder, 'points.txt')
    clusters_path = join(dst_folder, 'clusters.txt')

    # create directory
    Path(dst_folder).mkdir(parents=True, exist_ok=True)

    # load and write points
    img = cv2.imread(src_img).reshape((-1, 3)).astype(np.float32)
    with open(points_path, 'w') as f:
        f.write(nparray_to_str(img))
    print(f'Points saved in: {points_path}')

    # generate and save uniformly sampled centroids
    centroids = kmeans_plusplus_init(img, k)
    tmp_labels = np.arange(1, k + 1).reshape((k, 1))
    clusters = np.hstack((tmp_labels, centroids))

    with open(clusters_path, 'w') as f:
        f.write(nparray_to_str(clusters))
    print(f'Centroids saved in: {clusters_path}')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.src_img, args.dst_folder, args.k_init_centroids)
