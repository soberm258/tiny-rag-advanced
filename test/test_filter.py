import numpy as np
def __main__():
    distances = np.array([[5,4,0.2]], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.int64)
    print(distances, indices)
    max_distance = np.max(distances[0])
    threshold = max_distance * 0.2
    mask = distances >= threshold
    print("mask:", mask)
    distances = distances[mask]
    indices = indices[mask]
    print((distances, indices))
if __name__ == "__main__":
    __main__()