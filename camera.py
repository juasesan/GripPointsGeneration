import pyrealsense2 as rs
import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import seaborn as sns
import math


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
    color = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    #color = cv2.resize(color, (400, 400))
    _,thresh2 = cv2.threshold(color,100,255,cv2.THRESH_BINARY_INV)
    
    points = np.transpose(np.nonzero(thresh2))
    
    knn_graph = kneighbors_graph(points, 30, include_self=False)
    clustering = AgglomerativeClustering(n_clusters=30, linkage="ward", connectivity=knn_graph)
    clustering.fit(points)

    df = pd.DataFrame(points)
    df['cluster'] = clustering.labels_
    df.columns = ['Y', 'X', 'clusters']
    df['Y'] = df['Y'] * (-1)

    dfg = df.groupby('clusters').agg({'X':[np.mean], 'Y':[np.mean]}).reset_index()
    dfg.columns = ['cluster', 'X', 'Y']
    dfg['X'] = dfg['X'] / 400.0
    dfg['Y'] = (dfg['Y'] + 400.0) / 400.0
    dfg['Z'] = 0.0

    points_list = list(dfg[['X', 'Y']].itertuples(index=False, name=None))

    list_sorted = [points_list[0]]
    points_list.remove(points_list[0])
    node_points = []

    flag = True

    while flag:
        point1 = list_sorted[-1]
        distances = []
        points = []
        
        for point2 in points_list:
            distance = math.dist(point1, point2)
            distances.append(distance)
            points.append(point2)
            
        min_dist_1 = min(distances)
        min_idx_1 = distances.index(min_dist_1)
        
        distances.remove(min_dist_1)
        
        try:
            min_dist_2 = min(distances)
            min_idx_2 = distances.index(min_dist_2)

            factor = min_dist_2 / min_dist_1

            if (factor > 1) and (factor <= 1.3) and (list_sorted[-2] not in node_points) and (list_sorted[-3] not in node_points):
                node_points.append(point1)
                
        except:
            pass
        
        list_sorted.append(points[min_idx_1])
        points_list.remove(points[min_idx_1])
        
        if not points_list:
            flag = False

    df_points = pd.DataFrame(list_sorted, columns=['X', 'Y'])

    return df_points


def main():
    depth_img, color_img = take_picture()
    
    wire_points = get_points(color_img)
    #img = cv2.imread('./Data/sample images/1.jpg')
    #wire_points = get_points(img)

    print(wire_points)
    wire_points.to_csv('./points_xyz.csv')


if __name__ == '__main__':
    main()

        
