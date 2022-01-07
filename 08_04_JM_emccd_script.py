# 07/26/20 - Jose Mayorga
# 
# This script is designed to analyze images taken with an EMCCD camera. 
#
# Luminescent regions are selected based on predefined categories of mean luminescence values. 
# The regions are then cleaned, labeled and  filtered by predefined sizes as objects. The mean luminescence 
# value is then calculated by taking the mean of the sum of the mean luminescence values of each object. 
# The final value is adjusted based off either the min, mean or median luminescence value of the image.
# 
# Mean luminescence Value and Standard Deviation are reported.
#
# usage: python <script.py> <input_image.tiff>
#
# input_image - should be of '.tiff' format'

import sys
import skimage.io
import numpy as np
from scipy import ndimage
# from matplotlib import pyplot as plt # Uncomment plot statements to visualize process.

targetFile = sys.argv[1]

# Import image
image = skimage.io.imread(fname = targetFile)

# Copy image for final luminescence caluclation
cleanImage = image.copy()

# Create a mask that covers values less than a predefined luminescence level
if image.mean() < 10000: # Consider removing and simply reporting mean luminescence due to high sensitivity.
    mask = (image > image.min() + (image.mean() * 0.17)).astype(np.float)
elif image.mean() < 12500:
    mask = (image > image.max() * 0.4).astype(np.float)
elif image.mean() < 15000:
    mask = (image > image.max() * 0.5).astype(np.float)
elif image.mean() < 17000:
    mask = (image > image.max() * 0.6).astype(np.float)
else:
    mask = (image > image.max() * 0.7).astype(np.float)  

# Apply mask over image
image = mask + 0.1*np.random.randn(*mask.shape)
#plt.imshow(image)
binaryImg = image > 0.5 # Converts image into binary
#plt.imshow(binaryImg)

# Clean up luminescent objects
erodedImg = ndimage.binary_erosion(binaryImg)
reconstructImg = ndimage.binary_propagation(erodedImg, mask = binaryImg)
tmp = np.logical_not(reconstructImg)
erodedTmp = ndimage.binary_erosion(tmp)
reconstructFinal = np.logical_not(ndimage.binary_propagation(erodedTmp, mask = tmp))
#plt.imshow(reconstructFinal)

# Label regions to distinguish them
mask = (reconstructFinal > reconstructFinal.mean()).astype(np.float) # Recalulating mask
image = mask + 0.2*np.random.randn(*mask.shape)
labelIm, numLabels = ndimage.label(mask)
#numLabels # how many regions?
#plt.imshow(labelIm)

# Cacluate statistics: size of regions and mean luminescence of regions
sizes = ndimage.sum(mask, labelIm, range(numLabels + 1))
#meanVals = ndimage.sum(image, labelIm, range(1, numLabels + 1))

# Selecting based on maximum value of all sizes, since there is generally a large gap in size of luminescent objects 
if sizes.max() < 500:
    maskSize = sizes < sizes.max() * 0.3
elif sizes.max() < 1000:
    maskSize = sizes < sizes.max() * 0.25
elif sizes.max() < 2000:
    maskSize = sizes < sizes.max() * 0.09
else:
    maskSize = sizes < sizes.max() * 0.1

# Applying new mask
removePixels = maskSize[labelIm]
removePixels.shape
labelIm[removePixels] = 0
#plt.imshow(labelIm)        

# Relabeling image
labels = np.unique(labelIm)
labelIm = np.searchsorted(labels, labelIm)

# Saving stencil for manual observation
#plt.imshow(labelIm)  
#plt.savefig('07_31_stencil_w_image/stencil18.png')

# Calculate mean luminescence and standard deviation of objects
labelIm, numLabels = ndimage.label(labelIm)
meanLumVal = ((ndimage.mean(cleanImage, labelIm, range(1, numLabels + 1))).sum()/numLabels) - np.mean(cleanImage)
#meanLumVal = ((ndimage.mean(cleanImage, labelIm, range(1, numLabels + 1))).sum()/numLabels) - np.min(cleanImage)
#meanLumVal = ((ndimage.mean(cleanImage, labelIm, range(1, numLabels + 1))).sum()/numLabels) - np.median(cleanImage)

stdDeviation = np.std(ndimage.mean(cleanImage, labelIm, range(1, numLabels + 1)))

print('Mean Luminescence Value: ' + str(round(meanLumVal, 4)))
print('Standard Deviation: ' + str(round(stdDeviation, 4)))


#
