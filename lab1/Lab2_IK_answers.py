import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy.linalg import inv
from numpy import linalg as LA


class IK_Solver:
    def __init__(self, meta_data,joint_positions, joint_orientations, task2=False):
        self.meta_data = meta_data
        self.joint_offset = self.__compute_joint_offset()
        self.pose_data = self.__get_initial_pose()
        self.joint_name = self.meta_data.joint_name
        self.joint_parent = self.meta_data.joint_parent
        self.end_joint = self.meta_data.end_joint
        self.root_joint = self.meta_data.root_joint
        self.task2=task2
        self.joint_positions, self.joint_orientations = joint_positions, joint_orientations
        self.root_position = self.joint_positions[self.joint_name.index(
            self.root_joint)]  # self.meta_data.joint_initial_position[self.joint_name.index(self.root_joint)]


    def get_IK_result(self, target_pose, max_iters = 40):

        iter_count = 0
        alpha = 0.01
        joint_name = self.meta_data.joint_name
        joint_parent = self.meta_data.joint_parent
        end_joint = self.meta_data.end_joint
        path, path_name, path1, path2 = self.meta_data.get_path_from_root_to_end()
        error = 100.0
        while error > 0.01 and iter_count < max_iters:
            iter_count += 1
            # print(iter_count)
            joint_positions, _ = self.forward_kinematics()
            end_position = joint_positions[joint_name.index(end_joint)]
            delta = end_position - target_pose  # loss = 1/2*(delta*delya)
            J = self.__compute_Jacobian()
            gradient = J.transpose().dot(delta)
            Hessian = np.matmul(J.transpose(), J) + alpha * np.eye(self.pose_data.shape[0], dtype=np.float64)
            self.pose_data = self.pose_data - inv(Hessian).dot(gradient)
            end_position = joint_positions[joint_name.index(end_joint)]
            error = LA.norm(end_position - target_pose)
        return self.forward_kinematics()


    def get_error(self,target_pose):
        end_joint = self.meta_data.end_joint
        joint_positions, _ = self.forward_kinematics()
        joint_name = self.meta_data.joint_name
        end_position = joint_positions[joint_name.index(end_joint)]
        error = LA.norm(end_position - target_pose)
        return error

    def get_update_direction(self,target_pose):
        joint_name = self.meta_data.joint_name
        end_joint = self.meta_data.end_joint
        alpha = 0.01
        joint_positions, _ = self.forward_kinematics()
        end_position = joint_positions[joint_name.index(end_joint)]
        delta = end_position - target_pose  # loss = 1/2*(delta*delya)
        J = self.__compute_Jacobian()
        gradient = J.transpose().dot(delta)
        Hessian = np.matmul(J.transpose(), J) + alpha * np.eye(self.pose_data.shape[0], dtype=np.float64)
        d= - inv(Hessian).dot(gradient)
        return d

    def forward_kinematics(self):
        joint_name = self.meta_data.joint_name
        joint_parent = self.meta_data.joint_parent
        root_joint = self.meta_data.root_joint
        path, path_name, path1, path2 = self.meta_data.get_path_from_root_to_end()
        joint_positions = self.joint_positions.copy()
        joint_orientations = self.joint_orientations.copy()
        for i in range(len(joint_name)):
            joint_orientations[i][3] = 1.0
        # path2: new root to old root
        # path1: end joint to old root's child
        for i in range(len(path2)):
            idx = path2[i]
            joint = joint_name[idx]
            if joint == root_joint:
                joint_positions[idx] = self.root_position
                joint_orientations[idx] = (R.from_euler('XYZ', [0, 0, 0], degrees=False)).as_quat()
            else:
                prev_idx = path2[i - 1]

                joint_orientations[idx] = (R.from_quat(joint_orientations[prev_idx]) * R.from_euler('XYZ',
                                                                                                    [x for x in
                                                                                                     self.pose_data[
                                                                                                     idx * 3:idx * 3 + 3]],
                                                                                                    degrees=False)).as_quat()
                joint_positions[idx] = joint_positions[prev_idx] + R.from_quat(joint_orientations[idx]).apply(
                    -np.array(self.joint_offset[prev_idx]).astype(np.float64))

        joint_name_without_end=[]
        for name in joint_name:
            if not name.endswith('_end'):
                joint_name_without_end.append(name)
        for i in range(len(joint_name)):
            if i in path2:
                continue
            joint = joint_name[i]

            if joint.endswith('_end'):
                joint_orientations[i] = joint_orientations[joint_parent[i]]
                joint_positions[i] = joint_positions[joint_parent[i]] + R.from_quat(
                    joint_orientations[joint_parent[i]]).apply(self.joint_offset[i])
            else:
                joint_positions[i] = joint_positions[joint_parent[i]] + R.from_quat(
                    joint_orientations[joint_parent[i]]).apply(self.joint_offset[i])
                idx=joint_name_without_end.index(joint)
                joint_orientations[i] = (R.from_quat(joint_orientations[joint_parent[i]]) * R.from_euler('XYZ',
                                                                                                         self.pose_data[
                                                                                                         idx * 3:idx * 3 + 3],
                                                                                                         degrees=False)).as_quat()
        joint_positions, joint_orientations = np.array(joint_positions).astype(np.float64), joint_orientations
        if self.task2:
            for idx in path:
                self.joint_positions[idx], self.joint_orientations[idx]=joint_positions[idx], joint_orientations[idx]
            idx = joint_name.index('lWrist_end')
            self.joint_positions[idx]=joint_positions[joint_parent[idx]] + R.from_quat(
                        joint_orientations[joint_parent[idx]]).apply(self.joint_offset[idx])
            return self.joint_positions, self.joint_orientations
        else:
            return joint_positions, joint_orientations

    def __compute_Jacobian(self):
        # Geometric Approach
        joint_name = self.meta_data.joint_name
        joint_parent = self.meta_data.joint_parent
        root_joint = self.meta_data.root_joint
        end_joint = self.meta_data.end_joint
        path, path_name, path1, path2 = self.meta_data.get_path_from_root_to_end()
        joint_positions, _ = self.forward_kinematics()
        end_position = joint_positions[joint_name.index(end_joint)]
        Jacobian = []

        for i in range(len(joint_name)):
            joint = joint_name[i]
            if joint.endswith('_end'):
                continue
            if joint in path_name and joint != self.root_joint:
                for axis in range(3):
                    a = np.zeros([3], np.float64)
                    a[axis] = 1.0
                    r = end_position - joint_positions[i]
                    Jacobian.append(np.cross(a, r))
            else:
                for _ in range(3):
                    Jacobian.append(np.zeros([3], dtype=np.float64))

        Jacobian = np.array(Jacobian)
        for i in range(len(path2)):
            idx = path2[i]
            joint = joint_name[idx]
            if joint == root_joint:
                continue
        #print(Jacobian[0])
        Jacobian = np.array(Jacobian).astype(np.float64).transpose()
        return Jacobian

    def __get_initial_pose(self):
        joint_name = self.meta_data.joint_name
        ndim = 0
        for name in joint_name:
            if not name.endswith('_end'):
                ndim += 3
        return np.zeros([ndim], np.float64)

    def __compute_joint_offset(self):
        joint_name = self.meta_data.joint_name
        joint_parent = self.meta_data.joint_parent
        joint_initial_position = self.meta_data.joint_initial_position
        root_joint = self.meta_data.root_joint
        joint_offset = []
        for i in range(len(joint_name)):
            if joint_parent[i] == -1:
                joint_offset.append([0.0, 0.0, 0.0])
            else:
                joint_offset.append(joint_initial_position[i] - joint_initial_position[joint_parent[i]])
        return joint_offset


def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    # print(meta_data.joint_name)
    # print(path_name)
    # print(path)
    # print(path1)
    # print(path2)
    ik_solver = IK_Solver(meta_data,joint_positions, joint_orientations)
    joint_positions, joint_orientations = ik_solver.get_IK_result(target_pose)

    return joint_positions, joint_orientations


def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    #print(meta_data.joint_name)
    #print(path_name)
    #print(path)
    #print(path1)
    #print(path2)
    ik_solver = IK_Solver(meta_data,joint_positions,joint_orientations,True)
    target_pose = joint_positions[0] + np.array([relative_x, 0.0, relative_z]).astype(np.float64)
    target_pose[1] = target_height
    joint_positions, joint_orientations = ik_solver.get_IK_result(target_pose)

    return joint_positions, joint_orientations


def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    joint_name, joint_parent, joint_initial_position = meta_data.joint_name, meta_data.joint_parent, meta_data.joint_initial_position
    ik_solver0 = IK_Solver(meta_data, joint_positions, joint_orientations)
    ik_solver1 = IK_Solver(meta_data, joint_positions, joint_orientations)

    for _ in range(20):
        meta_data.end_joint = 'lWrist_end'
        ik_solver0.pose_data=ik_solver1.pose_data
        joint_positions, joint_orientations = ik_solver0.get_IK_result(left_target_pose,10)
        meta_data.end_joint = 'rWrist_end'
        ik_solver1.pose_data = ik_solver0.pose_data
        joint_positions, joint_orientations = ik_solver1.get_IK_result(right_target_pose,10)
    return joint_positions, joint_orientations
