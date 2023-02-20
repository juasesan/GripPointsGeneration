import pyrealsense2 as rs
import numpy as np
import rospy
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import math
import seaborn as sns


# Configure depth and color streams of the camera
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)


aligned_stream = rs.align(rs.stream.color) # alignment between color and depth
point_cloud = rs.pointcloud()
#blob_pub  = rospy.Publisher("/connector/position_connector",Point,queue_size=1)

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
    # Image pre-procesing
    color = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    color = cv2.resize(color, (320, 240))
    _,thresh2 = cv2.threshold(color,50,255,cv2.THRESH_BINARY_INV)
    
    points = np.transpose(np.nonzero(thresh2))
    
    # Clustering
    knn_graph = kneighbors_graph(points, 40, include_self=False)
    clustering = AgglomerativeClustering(n_clusters=40, linkage="ward", connectivity=knn_graph)
    clustering.fit(points)

    df = pd.DataFrame(points)
    df['cluster'] = clustering.labels_
    df.columns = ['Y', 'X', 'clusters']
    df['Y'] = df['Y'] * (-1)

    # Cable centroid points calculation
    cluster_df = df.groupby('clusters').agg({'X':[np.mean], 'Y':[np.mean]}).reset_index()
    cluster_df.columns = ['cluster', 'X', 'Y']

    points_list = list(cluster_df[['X', 'Y']].itertuples(index=False, name=None))

    for i in range(len(points_list)):       # Rounds each centroid point to the nearest point on its cluster
        cluster = df.loc[df['clusters'] == i]
        cluster_list = list(cluster[['X', 'Y']].itertuples(index=False, name=None))
        dist_list = []
        for pt in cluster_list:
            dist_list.append(math.dist(points_list[i], pt))
        idx = dist_list.index(min(dist_list))
        points_list[i] = cluster_list[idx]


    list_sorted = [points_list[0]]
    points_list.remove(points_list[0])
    
    # Sorting of the points to get the boundaries firts
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
        
        list_sorted.append(points[min_idx_1])
        points_list.remove(points[min_idx_1])
        
        if not points_list:
            flag = False

    distances_list = []
    for i in range(len(list_sorted)):
        if not i == (len(list_sorted) - 1):
            dist = math.dist(list_sorted[i], list_sorted[i+1])
            distances_list.append(dist)

    distances = pd.Series(distances_list.copy())
    threshold = distances.mean() + distances.std()
    first_pt_idx = distances_list.index(distances.loc[distances > threshold].iloc[0])

    list_sorted2 = [list_sorted[first_pt_idx]]
    list_sorted.remove(list_sorted[first_pt_idx])

    flag = True

    while flag:
        point1 = list_sorted2[-1]
        distances = []
        points = []
        
        for point2 in list_sorted:
            distance = math.dist(point1, point2)
            distances.append(distance)
            points.append(point2)
            
        min_idx_1 = distances.index(min(distances))
        
        list_sorted2.append(points[min_idx_1])
        list_sorted.remove(points[min_idx_1])
        
        if not list_sorted:
            flag = False
    
    df_points = pd.DataFrame(list_sorted2, columns=['X', 'Y'])

    # Coorditanes frame conversion, from camera frame to robot frame.
    center_x, center_y = 0.44, 0.9307   # Relative frame center
    
    df_points['Y'] = df_points['Y'] + 240.0
    df_points['Y'] = ((df_points['Y'] * 0.44) / 240) 
    df_points['Y'] = df_points['Y'] - center_y
    df_points['X'] = ((df_points['X'] * 0.82)/320)
    
    df_points['Z'] = 0.0

    return df_points


def main():
    depth_img, color_img = take_picture()
    
    wire_points = get_points(color_img)
    print("inicio")
    print(wire_points)
    fig,ax = plt.subplots(figsize=(15,15))
    plt.xlim(0, 320)
    plt.ylim(0, 240)
    sns.scatterplot(data=wire_points, x='X', y='Y', ax=ax)
    ax.imshow(color_img)
    plt.show()
    wire_points.to_csv('points_xyz.csv')


if __name__ == '__main__':
    main()