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

def rgb_to_cmyk(rgb):
    """Convert RGB to CMYK. Input is a numpy array of shape (N, 3)."""
    rgb = rgb / 255.0  # Normalize to [0, 1]
    K = 1 - np.max(rgb, axis=1, keepdims=True)
    C = (1 - rgb[:, 0:1] - K) / (1 - K + 1e-5)
    M = (1 - rgb[:, 1:2] - K) / (1 - K + 1e-5)
    Y = (1 - rgb[:, 2:3] - K) / (1 - K + 1e-5)
    CMYK = np.hstack((C, M, Y, K)) * 100  # Scale to percentage
    return CMYK

def kmeans_plus_plus(points, k):
    """KMeans++ initialization for selecting centroids."""
    centroids = []
    # Chọn ngẫu nhiên một điểm làm tâm đầu tiên
    centroids.append(points[np.random.randint(points.shape[0])])

    for _ in range(1, k):
        # Tính khoảng cách bình phương từ mỗi điểm đến tâm gần nhất
        distances = np.min([np.sum((points - c)**2, axis=1) for c in centroids], axis=0)
        # Tính xác suất tỷ lệ với khoảng cách bình phương
        probabilities = distances / np.sum(distances)
        # Lấy điểm tiếp theo dựa trên phân phối xác suất
        next_centroid_idx = np.random.choice(points.shape[0], p=probabilities)
        centroids.append(points[next_centroid_idx])

    return np.array(centroids)

def sample_points(points, sample_rate=0.1):
    """
    Lấy mẫu đều các điểm từ tập dữ liệu.
    :param points: Numpy array chứa các điểm (N, 4) (CMYK)
    :param sample_rate: Tỉ lệ lấy mẫu, mặc định là 10%.
    :return: Tập dữ liệu sau khi lấy mẫu.
    """
    total_points = len(points)
    sample_size = int(total_points * sample_rate)
    sampled_indices = np.linspace(0, total_points - 1, sample_size, dtype=int)
    return points[sampled_indices]

def nparray_to_str(X):
    lines = []
    for i in range(len(X)):
        line = ' '.join(str(X[i])[1:-1].split())
        if i == 0:
            line = line.replace('.', '')  # Bỏ dấu chấm ở dòng đầu tiên
        lines.append(line)
    return '\n'.join(lines)

def main(src_img, dst_folder, k, sample_rate=0.1):
    points_path = join(dst_folder, 'points.txt')
    clusters_path = join(dst_folder, 'clusters.txt')

    Path(dst_folder).mkdir(parents=True, exist_ok=True)

    # Load image and convert to CMYK
    img = cv2.imread(src_img).reshape((-1, 3)).astype(np.float32)
    cmyk_points = rgb_to_cmyk(img)

    # Lấy mẫu đều các điểm
    sampled_points = sample_points(cmyk_points, sample_rate=sample_rate)

    # Save CMYK points (cả tập dữ liệu gốc)
    with open(points_path, 'w') as f:
        f.write(nparray_to_str(cmyk_points))
    print(f'Points saved in: {points_path}')

    # Generate centroids using KMeans++ trên tập lấy mẫu
    selected_centroids = kmeans_plus_plus(sampled_points, k)
    tmp_labels = np.arange(1, k + 1).reshape((k, 1))
    clusters = np.hstack((tmp_labels, selected_centroids))

    # Save centroids
    with open(clusters_path, 'w') as f:
        f.write(nparray_to_str(clusters))
    print(f'Centroids saved in: {clusters_path}')

if __name__ == '__main__':
    args = parser.parse_args()
    main("/home/vu/kmeans_mapreduce-master/image1.jpg", "/home/vu/kmeans_mapreduce-master", 20)