import subprocess
import pyautogui
import open3d as o3d
import numpy as np
import time
import imageio
import os
import glob 

# Capture the window contents as screenshots
def capture_screenshot(file_path):
    screenshot = pyautogui.screenshot()
    screenshot.save(file_path)

if __name__ == '__main__':
    # Specify the directory and group names
    data_dir = './AzureKinectRecord_30_05/'
    group_names = ['1', '2', '3']

    # Specify the time t and duration to show each point cloud
    time_t = 0.1  # Replace with the desired time t
    duration_secs = 0.05  # Replace with the desired duration in seconds

    # Specify video output settings
    fps = 10  # Frames per second

    for group_name in group_names:

        print(f"\n Current group is {group_name} \n")
        # Load and visualize point clouds for all frames at time t
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window(width=1920, height=1080)

        # Create a folder to store the rendered frames
        frames_folder = f'{data_dir}/{group_name}/point_cloud_frames_group'
        os.makedirs(frames_folder, exist_ok=True)

        total_frames = len(glob.glob("%s/%s/azure_kinect1_2/color/color*.jpg" % (data_dir, group_name)))
        print('number of images %i' % total_frames)

        if group_name == '3':
            start_index = 89
        else:
            start_index = 0

        for frame_idx in range(start_index, start_index + total_frames):  # Replace 'total_frames' with the total number of frames in the series
            # Generate the file path for the current frame
            save_name = "%s/%s/point_cloud/pose%04i.ply" % (data_dir, group_name, frame_idx)

            # Load the point cloud from file
            point_cloud = o3d.io.read_point_cloud(save_name)

            # Flip the point cloud along the Z-axis
            point_cloud.points = o3d.utility.Vector3dVector(np.asarray(point_cloud.points) * np.array([1, -1, -1]))

            # Customize the visualization settings
            visualizer.add_geometry(point_cloud)
            visualizer.get_render_option().background_color = np.asarray([0, 0, 0])  # Set background color to black
            visualizer.get_render_option().point_size = 1.0  # Set point size
            visualizer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))  # Add coordinate frame
            visualizer.update_geometry(point_cloud)
            visualizer.poll_events()
            visualizer.update_renderer()

            # Save the current frame as an image
            capture_screenshot("%s/%s/point_cloud_frames_group/frame_%04d.png" % (data_dir, group_name, frame_idx))
            # image_path = os.path.join(frames_folder, f'frame_{frame_idx:04d}.png')
            # visualizer.capture_screen_image(image_path)

            # Wait for the specified duration
            time.sleep(duration_secs)

            # Clear the previous point cloud from the visualization
            visualizer.remove_geometry(point_cloud)

        # Convert the rendered frames into a video using external tools (e.g., FFmpeg)
        video_output_path = f'{data_dir}/{group_name}/point_cloud_animation_group_{group_name}.mp4'
        cmd = f"ffmpeg -r {fps} -start_number {start_index} -i {frames_folder}/frame_%04d.png -c:v libx264 -vf fps={fps} -pix_fmt yuv420p {video_output_path}"
        os.system(cmd)

        # Delete the individual image files
        subprocess.run('del point_cloud_frames_group_%s\*.png' % group_name, shell=True)

        # Destroy the visualization window
        visualizer.destroy_window()