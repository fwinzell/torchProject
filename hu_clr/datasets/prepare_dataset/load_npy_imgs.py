import numpy as np
import os
import fnmatch

root_dir = "/home/fi5666wi/Documents/Task04_Hippocampus/orig"
fls = []
for root, dirs, files in os.walk(root_dir):
    for filename in sorted(fnmatch.filter(files, "*.npy")):
        npy_file = os.path.join(root, filename)
        numpy_array = np.load(npy_file, mmap_mode="r+")  # change "r" to "r+"
        img = numpy_array[:,0,:,:]
        tar = numpy_array[:,1,:,:]

        fls.append(npy_file)

print(len(fls))