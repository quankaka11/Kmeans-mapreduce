#!/usr/bin/env python3

import sys

def main():
    current_cid = None
    current_sum = None
    current_count = 0

    for line in sys.stdin:
        cid, point_str = line.strip().split("\t")
        point = list(map(float, point_str.split()))

        if current_cid is None:
            current_cid = cid
            current_sum = point
            current_count = 1
        elif cid == current_cid:
            current_sum = [s + p for s, p in zip(current_sum, point)]
            current_count += 1
        else:
            # Emit new centroid
            new_centroid = [s / current_count for s in current_sum]
            print(f"{current_cid} {' '.join(map(str, new_centroid))}")

            # Reset for next cluster
            current_cid = cid
            current_sum = point
            current_count = 1

    # Handle last cluster
    if current_cid is not None:
        new_centroid = [s / current_count for s in current_sum]
        print(f"{current_cid} {' '.join(map(str, new_centroid))}")

if __name__ == "__main__":
    main()