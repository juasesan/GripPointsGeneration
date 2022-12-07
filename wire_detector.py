import numpy as np
import cv2 as cv
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import pyrealsense2 as rs

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Wait for a coherent pair of frames: depth and color
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

# Convert images to numpy arrays
depth_image = np.asanyarray(depth_frame.get_data())
color_image = np.asanyarray(color_frame.get_data())

# Apply colormap on depth image (image must be converted to 8-bit per pixel first)
depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)

depth_colormap_dim = depth_colormap.shape
color_colormap_dim = color_image.shape

# If depth and color resolutions are different, resize color image to match depth image for display
resized_color_image = cv.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv.INTER_AREA)

# Stop streaming
pipeline.stop()


#img = cv.imread('E:/Github projects/wire_harness_segmentation/Data/sample images/3.jpg')
img = cv.cvtColor(resized_color_image, cv.COLOR_BGR2GRAY)
img = cv.resize(img, (400, 400))

plt.subplots(figsize=(10,8)),plt.imshow(img, 'gray', vmin=0, vmax=255)
plt.show()

_,thresh2 = cv.threshold(img,100,255,cv.THRESH_BINARY_INV)

plt.subplots(figsize=(10,8)),plt.imshow(thresh2, 'gray')
plt.show()

points = np.transpose(np.nonzero(thresh2))

clustering = AgglomerativeClustering(n_clusters=40, linkage="average")
clustering.fit(points)

df = pd.DataFrame(points)
df['cluster'] = clustering.labels_
df.columns = ['Y', 'X', 'clusters']
df['Y'] = df['Y'] * (-1)

fig,ax = plt.subplots(figsize=(15,12))
sns.scatterplot(data=df, x="X", y='Y', hue="clusters", ax=ax)
_=ax.set_xlim(0,400)
_=ax.set_ylim(-400,0)
plt.show()

dfg = df.groupby('clusters').agg({'X':[np.mean], 'Y':[np.mean]}).reset_index()
dfg.columns = ['cluster', 'X_cent', 'Y_cent']

fig,ax = plt.subplots(figsize=(15,12))
sns.scatterplot(data=dfg, x="X_cent", y='Y_cent', ax=ax)
#_=ax.set_xlim(0,400)
#_=ax.set_ylim(-400,0)
plt.xlim(0, 400)
plt.ylim(-400, 0)
ax.imshow(thresh2, extent=[0, 400, -400, 0], aspect='auto', alpha=0.5)
plt.show()

