# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import rospy
import tf
import open3d as o3d
from collections import deque
from os import listdir
from os.path import join
from sensor_msgs.msg import Image, CameraInfo
from open3d_interface.srv import StartRecording, StopRecording
from open3d_interface.srv import StartRecordingResponse, StopRecordingResponse
from open3d_interface.utility.file import read_pose
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TransformStamped

# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

tf_listener = None
camera_info_topic = '/camera/rgb/camera_info'
current_index = 0
publishing = False


def startPublishingCallback(req):
    global publishing
    publishing = True

    return StartRecordingResponse(True)


def stopPublishingCallback(req):
    global publishing
    publishing = False

    return StopRecordingResponse(True)

def restartPublishingCallback(req):
    global publishing, current_index
    current_index = 0

    return StartRecordingResponse(True)


def main():
    global camera_info_topic, tf_listener, publishing, current_index

    rospy.init_node('open3d_dir_2_pub', anonymous=True)

    # Get parameters
    depth_image_topic = rospy.get_param('~depth_image_topic')
    color_image_topic = rospy.get_param('~color_image_topic')
    camera_info_topic = rospy.get_param('~camera_info_topic')
    pose_rel_frame = rospy.get_param('~relative_frame')
    pose_track_frame = rospy.get_param('~tracking_frame')
    img_dir = rospy.get_param('~image_directory')
    pub_rate = rospy.get_param('~pub_rate')

    color_dir = join(img_dir, 'color')
    depth_dir = join(img_dir, 'depth')
    pose_dir = join(img_dir, 'pose')
    camera_intrinsic_fp = join(img_dir, 'camera_intrinsic.json')
    intrinsic = o3d.io.read_pinhole_camera_intrinsic(camera_intrinsic_fp)

    camera_intrinsic_msg = CameraInfo()
    camera_intrinsic_msg.header.seq = 0
    camera_intrinsic_msg.header.stamp = rospy.get_rostime()
    camera_intrinsic_msg.header.frame_id = pose_track_frame
    camera_intrinsic_msg.height = intrinsic.height
    camera_intrinsic_msg.width = intrinsic.width
    camera_intrinsic_msg.K = [intrinsic.intrinsic_matrix[0][0], 0, intrinsic.intrinsic_matrix[0][2],
                              0, intrinsic.intrinsic_matrix[1][1], intrinsic.intrinsic_matrix[1][2],
                              0, 0, intrinsic.intrinsic_matrix[2][2]]

    br = tf.TransformBroadcaster()

    num_imgs = len(listdir(color_dir))

    rospy.loginfo(rospy.get_caller_id() + ": depth_image_topic - " + depth_image_topic)
    rospy.loginfo(rospy.get_caller_id() + ": color_image_topic - " + color_image_topic)
    rospy.loginfo(rospy.get_caller_id() + ": camera_info_topic - " + camera_info_topic)

    depth_pub = rospy.Publisher(depth_image_topic, Image, queue_size=10)
    rgb_pub = rospy.Publisher(color_image_topic, Image, queue_size=10)
    camera_info_pub = rospy.Publisher(camera_info_topic, CameraInfo, queue_size=10)

    start_server = rospy.Service('start_publishing', StartRecording, startPublishingCallback)
    stop_server = rospy.Service('stop_publishing', StopRecording, stopPublishingCallback)
    restart_server = rospy.Service('restart_publishing', StartRecording, restartPublishingCallback)

    rate = rospy.Rate(pub_rate)
    bridge = CvBridge()

    tf_stamped = TransformStamped()
    tf_stamped.header.seq = 0
    tf_stamped.header.stamp = rospy.get_rostime()
    tf_stamped.header.frame_id = pose_rel_frame
    tf_stamped.child_frame_id = pose_track_frame

    print("Looping")
    while not rospy.is_shutdown():
        if publishing:
            current_index += 1
            if current_index >= num_imgs:
                current_index = 0
            color_index_string = f"{current_index:06d}" + ".jpg"
            depth_index_string = f"{current_index:06d}" + ".png"
            pose_index_string = f"{current_index:06d}" + ".pose"
            color_img_fp = join(color_dir, color_index_string)
            depth_img_fp = join(depth_dir, depth_index_string)
            pose_fp = join(pose_dir, pose_index_string)
            color_img = o3d.io.read_image(color_img_fp)
            depth_img = o3d.io.read_image(depth_img_fp)
            pose = read_pose(pose_fp)
            np_pose = np.asarray(pose)
            rotation = R.from_dcm(np_pose[0:3, 0:3])
            quat = rotation.as_quat()
            curr_time = rospy.get_rostime()
            br.sendTransform(np_pose[0:3, 3],
                             quat,
                             curr_time,
                             pose_track_frame,
                             pose_rel_frame)
            print("Current index:", current_index)
            camera_intrinsic_msg.header.stamp = curr_time
            camera_info_pub.publish(camera_intrinsic_msg)
            image_message_color = bridge.cv2_to_imgmsg(np.asarray(color_img), encoding='bgr8')
            image_message_color.header.stamp = curr_time
            image_message_depth = bridge.cv2_to_imgmsg(np.asarray(depth_img), encoding='16UC1')
            image_message_depth.header.stamp = curr_time
            rgb_pub.publish(image_message_color)
            depth_pub.publish(image_message_depth)
        rate.sleep()
    cv2.destroyAllWindows()
    print("Done Looping")