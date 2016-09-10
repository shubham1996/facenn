# Util imports, from the standard library
import os
import sys
import time as time

# Imports from numpy, scipy and sklearn libraries to aid PCA
import numpy
from scipy import misc
from sklearn import decomposition

# Imports for plotting
import matplotlib.pyplot as plt

# Terminate if program arguments are invalid
if sys.argv.__len__() != 2 :
    exit(1)
person = sys.argv[1]

# Method to plot the obtained eigenfaces in a gallery-like matrix
def plot_gallery(images, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(images.shape[0]):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())

# Get a sorted list of available PGM files for the specified person
files = os.listdir('orl_faces/'+person+'/')
files.sort()

# Get the 2D matrix representation of the image, flatten it in 1D and stack it vertically on each other
images = numpy.vstack(misc.imread('orl_faces/'+person+'/'+file).flatten() for file in files)

# Run the PCA algorithm on the generated matrix
pca = decomposition.PCA()
pca.fit(images)
eigenfaces = pca.components_.reshape((10,112, 92))

# Show the eigenface plots
plot_gallery(eigenfaces, 112, 92)
plt.show()

# Wait for termination
input("Hit enter to terminate")