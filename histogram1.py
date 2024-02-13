import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms

img1 = cv2.imread('assignment1.jpg')
print('No of Channel is: ' + str(img1.shape[2]))

img2 = cv2.imread('racecar2.jpg')
print('No of Channel is: ' + str(img2.shape[2]))

# Split images into individual channels
img1_channels = cv2.split(img1)
img2_channels = cv2.split(img2)

matched_channels = []

# Perform histogram matching for each channel
for i in range(img1.shape[2]):
    matched_channel = match_histograms(img1_channels[i], img2_channels[i])
    matched_channels.append(matched_channel)

# Convert matched_channels list to a NumPy array
matched = cv2.merge(matched_channels)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)

for aa in (ax1, ax2, ax3):
    aa.set_axis_off()

ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
ax1.set_title('Source')
ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
ax2.set_title('Reference')
ax3.imshow(matched)
ax3.set_title('Matched')

plt.tight_layout()
plt.show()