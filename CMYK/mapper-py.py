#!/usr/bin/env python3

import sys
import math

centroids = []

def load_centroids():
    """Load centroids from clusters.txt"""
    global centroids
    with open('clusters.txt', 'r') as f:
        for line in f:
            parts = line.strip().split()
            centroids.append((int(parts[0]), list(map(float, parts[1:]))))

def closest_centroid(point):
    """Find the closest centroid to a given point."""
    min_distance = float('inf')
    closest_id = -1
    for cid, ccoords in centroids:
        distance = math.sqrt(sum((p - c) ** 2 for p, c in zip(point, ccoords)))
        if distance < min_distance:
            min_distance = distance
            closest_id = cid
    return closest_id

def main():
    load_centroids()
    for line in sys.stdin:
        point = list(map(float, line.strip().split()))
        cid = closest_centroid(point)
        print(f"{cid}\t{' '.join(map(str, point))}")  # Emit <CentroidID, Point>

if __name__ == "__main__":
    main()