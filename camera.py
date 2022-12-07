import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)


# Start streaming
pipeline.start(config)

try:
    while True:
        input("Press Enter to continue...")

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)

        img = cv2.cvtColor(resized_color_image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (400, 400))
        
        plt.subplots(figsize=(10,8)),plt.imshow(img, 'gray', vmin=0, vmax=255)
        plt.show()

finally:

    # Stop streaming
    pipeline.stop()