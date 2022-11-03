import numpy as np
from scipy.spatial.transform import Rotation as R


def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i + 1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1, -1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.split()[0] == 'JOINT' or line.split()[0] == 'ROOT':
                joint_name.append(line.split()[1])
            elif line.split()[0] == 'End':
                joint_name.append(joint_name[-1] + '_end')
        for line in lines:
            if line.split()[0] == 'OFFSET':
                joint_offset.append([line.split()[k] for k in range(1, 4)])
        id_dict = {}
        for i in range(len(joint_name)):
            id_dict[joint_name[i]] = i
        current_parent = [-1, ]
        current_id = -1
        for line in lines:
            if line.split()[0] == 'JOINT' or line.split()[0] == 'ROOT' or line.split()[0] == 'End':
                joint_parent.append(current_parent[-1])
                current_id += 1
            elif line.split()[0] == '{':
                current_parent.append(current_id)
            elif line.split()[0] == '}':
                current_parent.pop()
        # print(joint_name)
    return joint_name, joint_parent, np.array(joint_offset).astype(np.float64)


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
    """
    joint_positions = []
    joint_orientations = []
    pose_data = motion_data[frame_id]
    root_offset = pose_data[0:3]
    rotationMatrix = R.from_euler('XYZ', pose_data[3:6], degrees=True)
    joint_orientations.append(rotationMatrix)
    joint_positions.append(rotationMatrix.apply(joint_offset[0]) + root_offset)
    pose_data = pose_data[6:]

    for i in range(1, len(joint_name)):
        if joint_name[i].endswith('_end'):
            joint_orientations.append(joint_orientations[joint_parent[i]])

            joint_positions.append(
                joint_positions[joint_parent[i]] + joint_orientations[joint_parent[i]].apply(joint_offset[i]))
        else:
            local_rotation = R.from_euler('XYZ', pose_data[0:3], degrees=True)
            pose_data = pose_data[3:]
            joint_orientations.append(joint_orientations[joint_parent[i]] * local_rotation)

            joint_positions.append(
                joint_positions[joint_parent[i]] + joint_orientations[joint_parent[i]].apply(joint_offset[i]))

    return np.array(joint_positions).astype(np.float64), np.array([m.as_quat() for m in joint_orientations]).astype(
        np.float64)


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
    """
    T_order = ['RootJoint', 'lHip', 'lKnee', 'lAnkle', 'lToeJoint', 'lToeJoint_end', 'pelvis_lowerback',
               'lowerback_torso', 'lTorso_Clavicle', 'lShoulder', 'lElbow', 'lWrist', 'lWrist_end', 'rTorso_Clavicle',
               'rShoulder', 'rElbow', 'rWrist', 'rWrist_end', 'torso_head', 'torso_head_end', 'rHip', 'rKnee', 'rAnkle',
               'rToeJoint', 'rToeJoint_end']
    A_order = ['RootJoint', 'lHip', 'lKnee', 'lAnkle', 'lToeJoint', 'lToeJoint_end', 'rHip', 'rKnee', 'rAnkle',
               'rToeJoint', 'rToeJoint_end', 'pelvis_lowerback', 'lowerback_torso', 'torso_head', 'torso_head_end',
               'lTorso_Clavicle', 'lShoulder', 'lElbow', 'lWrist', 'lWrist_end', 'rTorso_Clavicle', 'rShoulder',
               'rElbow', 'rWrist', 'rWrist_end']
    T_id_dict = {}
    A_id_dict = {}
    for i in range(len(T_order)):
        T_id_dict[T_order[i]] = i
        A_id_dict[A_order[i]] = i
    A_motion_data = load_motion_data(A_pose_bvh_path).tolist()
    T_motion_data = []
    for pose_data in A_motion_data:
        A_elements = []
        for a_joint in A_order:
            if a_joint == 'RootJoint':
                A_elements.append(pose_data[0:6])
                pose_data=pose_data[6:]
            elif a_joint.endswith('_end'):
                A_elements.append([])
            elif a_joint == 'rShoulder':
                temp=pose_data[0:3]
                temp[2]+=45.0
                A_elements.append(temp)
                pose_data = pose_data[3:]
            elif a_joint == 'lShoulder':
                temp=pose_data[0:3]
                temp[2]-=45.0
                A_elements.append(temp)
                pose_data = pose_data[3:]
            else:
                A_elements.append(pose_data[0:3])
                pose_data=pose_data[3:]
        T_elements=[]
        for t_joint in T_order:
            T_elements.append(A_elements[A_id_dict[t_joint]])
        T_pose_data = []
        for e in T_elements:
            T_pose_data=T_pose_data+e
        T_motion_data.append(T_pose_data)

    motion_data = np.array(T_motion_data).astype(np.float64)
    return motion_data
