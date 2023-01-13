import pyrealsense2 as rs
import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns


# Configure depth and color streams of the camera
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)


aligned_stream = rs.align(rs.stream.color) # alignment between color and depth
point_cloud = rs.pointcloud()

# Start streaming
profile = pipeline.start(config)

def take_picture():
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    color_image = cv2.resize(color_image, dsize=(depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_AREA)

    return depth_image, color_image


def get_points(color, depth=None):
    _,thresh2 = cv2.threshold(color,100,255,cv2.THRESH_BINARY_INV)
    
    points = np.transpose(np.nonzero(thresh2))
    
    clustering = AgglomerativeClustering(n_clusters=40, linkage="average")
    clustering.fit(points)

    df = pd.DataFrame(points)
    df['cluster'] = clustering.labels_
    df.columns = ['Y', 'X', 'clusters']
    df['Y'] = df['Y'] * (-1)

    dfg = df.groupby('clusters').agg({'X':[np.mean], 'Y':[np.mean]}).reset_index()
    dfg.columns = ['cluster', 'X', 'Y']
    dfg['Z'] = 0.0

    return dfg


def main():
    depth_img, color_img = take_picture()
    
    wire_points = get_points(color_img)

    print(wire_points)
    wire_points.to_csv('./points_xyz.csv')


if __name__ == '__main__':
    main()

        
