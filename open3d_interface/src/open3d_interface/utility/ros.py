# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import json
import open3d as o3d
import numpy as np
from geometry_msgs.msg import Point, Vector3, Pose
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

def save_camera_info_intrinsic_as_json(filename, camera_info_msg):
    with open(filename, 'w') as outfile:
        obj = json.dump(
            {
                'width':
                    camera_info_msg.width,
                'height':
                    camera_info_msg.height,
                'intrinsic_matrix': [
                    camera_info_msg.k[0], 0, 0, 0, camera_info_msg.k[4], 0, camera_info_msg.k[2],
                    camera_info_msg.k[5], 1
                ]
            },
            outfile,
            indent=4)


def getIntrinsicsFromMsg(camera_info_msg):
  return o3d.camera.PinholeCameraIntrinsic(camera_info_msg.width, camera_info_msg.height, camera_info_msg.k[0], camera_info_msg.k[4], camera_info_msg.k[2], camera_info_msg.k[5])


def transformStampedToVectors(gm_tf_stamped):
    vec_t = gm_tf_stamped.transform.translation
    vec_q = gm_tf_stamped.transform.rotation
    translation = np.array([vec_t.x, vec_t.y, vec_t.z])
    quaternion = np.array([vec_q.w, vec_q.x, vec_q.y, vec_q.z])
    return translation, quaternion


def meshToRos(mesh):
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    vertex_colors = np.asarray(mesh.vertex_colors)
    out_msg = Marker()
    out_msg.type = out_msg.TRIANGLE_LIST
    out_msg.action = out_msg.ADD
    out_msg.id = 1
    out_msg.scale.x = 1.0
    out_msg.scale.y = 1.0
    out_msg.scale.z = 1.0
    out_msg.pose.position.x = 0.0
    out_msg.pose.position.y = 0.0
    out_msg.pose.position.z = 0.0
    out_msg.pose.orientation.w = 1.0
    out_msg.pose.orientation.x = 0.0
    out_msg.pose.orientation.y = 0.0
    out_msg.pose.orientation.z = 0.0
    for triangle in triangles:
        for vertex_index in triangle:
            curr_point = Point()
            curr_point.x = vertices[vertex_index][0]
            curr_point.y = vertices[vertex_index][1]
            curr_point.z = vertices[vertex_index][2]
            curr_point_color = ColorRGBA()
            curr_point_color.r = vertex_colors[vertex_index][0]
            curr_point_color.g = vertex_colors[vertex_index][1]
            curr_point_color.b = vertex_colors[vertex_index][2]
            curr_point_color.a = 1.0
            out_msg.points.append(curr_point)
            out_msg.colors.append(curr_point_color)
    return out_msg
