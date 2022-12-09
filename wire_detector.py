import numpy as np
import cv2 as cv
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import pyrealsense2 as rs


img = cv.imread('E:/Github projects/wire_harness_segmentation/Data/sample images/3.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
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
dfg.to_csv('points.csv')

fig,ax = plt.subplots(figsize=(15,12))
sns.scatterplot(data=dfg, x="X_cent", y='Y_cent', ax=ax)
#_=ax.set_xlim(0,400)
#_=ax.set_ylim(-400,0)
plt.xlim(0, 400)
plt.ylim(-400, 0)
ax.imshow(thresh2, extent=[0, 400, -400, 0], aspect='auto', alpha=0.5)
plt.show()

