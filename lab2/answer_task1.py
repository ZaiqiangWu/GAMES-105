import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


# ------------- lab1里的代码 -------------#
def load_meta_data(bvh_path):
    with open(bvh_path, 'r') as f:
        channels = []
        joints = []
        joint_parents = []
        joint_offsets = []
        end_sites = []

        parent_stack = [None]
        for line in f:
            if 'ROOT' in line or 'JOINT' in line:
                joints.append(line.split()[-1])
                joint_parents.append(parent_stack[-1])
                channels.append('')
                joint_offsets.append([0, 0, 0])

            elif 'End Site' in line:
                end_sites.append(len(joints))
                joints.append(parent_stack[-1] + '_end')
                joint_parents.append(parent_stack[-1])
                channels.append('')
                joint_offsets.append([0, 0, 0])

            elif '{' in line:
                parent_stack.append(joints[-1])

            elif '}' in line:
                parent_stack.pop()

            elif 'OFFSET' in line:
                joint_offsets[-1] = np.array([float(x) for x in line.split()[-3:]]).reshape(1, 3)

            elif 'CHANNELS' in line:
                trans_order = []
                rot_order = []
                for token in line.split():
                    if 'position' in token:
                        trans_order.append(token[0])

                    if 'rotation' in token:
                        rot_order.append(token[0])

                channels[-1] = ''.join(trans_order) + ''.join(rot_order)

            elif 'Frame Time:' in line:
                break

    joint_parents = [-1] + [joints.index(i) for i in joint_parents[1:]]
    channels = [len(i) for i in channels]
    return joints, joint_parents, channels, joint_offsets


def load_motion_data(bvh_path):
    with open(bvh_path, 'r') as f:
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


# ------------- 实现一个简易的BVH对象，进行数据处理 -------------#

'''
注释里统一N表示帧数，M表示关节数
position, rotation表示局部平移和旋转
translation, orientation表示全局平移和旋转
'''


class BVHMotion():
    def __init__(self, bvh_file_name=None) -> None:

        # 一些 meta data
        self.joint_name = []
        self.joint_channel = []
        self.joint_parent = []

        # 一些local数据, 对应bvh里的channel, XYZposition和 XYZrotation
        # ! 这里我们把没有XYZ position的joint的position设置为offset, 从而进行统一
        self.joint_position = None  # (N,M,3) 的ndarray, 局部平移
        self.joint_rotation = None  # (N,M,4)的ndarray, 用四元数表示的局部旋转

        if bvh_file_name is not None:
            self.load_motion(bvh_file_name)
        pass

    # ------------------- 一些辅助函数 ------------------- #
    def load_motion(self, bvh_file_path):
        '''
            读取bvh文件，初始化元数据和局部数据
        '''
        self.joint_name, self.joint_parent, self.joint_channel, joint_offset = \
            load_meta_data(bvh_file_path)

        motion_data = load_motion_data(bvh_file_path)

        # 把motion_data里的数据分配到joint_position和joint_rotation里
        self.joint_position = np.zeros((motion_data.shape[0], len(self.joint_name), 3))
        self.joint_rotation = np.zeros((motion_data.shape[0], len(self.joint_name), 4))
        self.joint_rotation[:, :, 3] = 1.0  # 四元数的w分量默认为1

        cur_channel = 0
        for i in range(len(self.joint_name)):
            if self.joint_channel[i] == 0:
                self.joint_position[:, i, :] = joint_offset[i].reshape(1, 3)
                continue
            elif self.joint_channel[i] == 3:
                self.joint_position[:, i, :] = joint_offset[i].reshape(1, 3)
                rotation = motion_data[:, cur_channel:cur_channel + 3]
            elif self.joint_channel[i] == 6:
                self.joint_position[:, i, :] = motion_data[:, cur_channel:cur_channel + 3]
                rotation = motion_data[:, cur_channel + 3:cur_channel + 6]
            self.joint_rotation[:, i, :] = R.from_euler('XYZ', rotation, degrees=True).as_quat()
            cur_channel += self.joint_channel[i]

        return

    def batch_forward_kinematics(self, joint_position=None, joint_rotation=None):
        '''
        利用自身的metadata进行批量前向运动学
        joint_position: (N,M,3)的ndarray, 局部平移
        joint_rotation: (N,M,4)的ndarray, 用四元数表示的局部旋转
        '''
        if joint_position is None:
            joint_position = self.joint_position
        if joint_rotation is None:
            joint_rotation = self.joint_rotation

        joint_translation = np.zeros_like(joint_position)
        joint_orientation = np.zeros_like(joint_rotation)
        joint_orientation[:, :, 3] = 1.0  # 四元数的w分量默认为1

        # 一个小hack是root joint的parent是-1, 对应最后一个关节
        # 计算根节点时最后一个关节还未被计算，刚好是0偏移和单位朝向

        for i in range(len(self.joint_name)):
            pi = self.joint_parent[i]
            parent_orientation = R.from_quat(joint_orientation[:, pi, :])
            joint_translation[:, i, :] = joint_translation[:, pi, :] + \
                                         parent_orientation.apply(joint_position[:, i, :])
            joint_orientation[:, i, :] = (parent_orientation * R.from_quat(joint_rotation[:, i, :])).as_quat()
        return joint_translation, joint_orientation

    def adjust_joint_name(self, target_joint_name):
        '''
        调整关节顺序为target_joint_name
        '''
        idx = [self.joint_name.index(joint_name) for joint_name in target_joint_name]
        self.joint_name = [self.joint_name[i] for i in idx]
        self.joint_parent = [self.joint_parent[i] for i in idx]
        self.joint_channel = [self.joint_channel[i] for i in idx]
        self.joint_position = self.joint_position[:, idx, :]
        self.joint_rotation = self.joint_rotation[:, idx, :]
        pass

    def raw_copy(self):
        '''
        返回一个拷贝
        '''
        return copy.deepcopy(self)

    @property
    def motion_length(self):
        return self.joint_position.shape[0]

    def sub_sequence(self, start, end):
        '''
        返回一个子序列
        start: 开始帧
        end: 结束帧
        '''
        res = self.raw_copy()
        res.joint_position = res.joint_position[start:end, :, :]
        res.joint_rotation = res.joint_rotation[start:end, :, :]
        return res

    def append(self, other):
        '''
        在末尾添加另一个动作
        '''
        other = other.raw_copy()
        other.adjust_joint_name(self.joint_name)
        self.joint_position = np.concatenate((self.joint_position, other.joint_position), axis=0)
        self.joint_rotation = np.concatenate((self.joint_rotation, other.joint_rotation), axis=0)
        pass

    # --------------------- 你的任务 -------------------- #

    def decompose_rotation_with_yaxis(self, rotation):
        '''
        输入: rotation 形状为(4,)的ndarray, 四元数旋转
        输出: Ry, Rxz，分别为绕y轴的旋转和转轴在xz平面的旋转，并满足R = Ry * Rxz
        '''
        Ry = np.zeros_like(rotation)
        Rxz = np.zeros_like(rotation)
        # TODO: 你的代码
        # https://blog.csdn.net/lyyiangang/article/details/121929006
        Rxyz = R.from_quat(rotation)
        euler_yxz = Rxyz.as_euler('YXZ', True)
        Ry = R.from_euler('Y', euler_yxz[0], True).as_quat()
        Rxz = R.from_euler('XZ', [euler_yxz[1], euler_yxz[2]], True).as_quat()
        return Ry, Rxz

    # part 1
    def translation_and_rotation(self, frame_num, target_translation_xz, target_facing_direction_xz):
        '''
        计算出新的joint_position和joint_rotation
        使第frame_num帧的根节点平移为target_translation_xz, 水平面朝向为target_facing_direction_xz
        frame_num: int
        target_translation_xz: (2,)的ndarray
        target_faceing_direction_xz: (2,)的ndarray，表示水平朝向。你可以理解为原本的z轴被旋转到这个方向。
        Tips:
            主要是调整root节点的joint_position和joint_rotation
            frame_num可能是负数，遵循python的索引规则
            你需要完成并使用decompose_rotation_with_yaxis
            输入的target_facing_direction_xz的norm不一定是1
        '''
        #print(frame_num)

        res = self.raw_copy()  # 拷贝一份，不要修改原始数据

        # 比如说，你可以这样调整第frame_num帧的根节点平移
        offset = target_translation_xz - res.joint_position[frame_num, 0, [0, 2]]
        res.joint_position[:, 0, [0, 2]] += offset
        # TODO: 你的代码

        target_vector = np.array([target_facing_direction_xz[0], 0, target_facing_direction_xz[1]], np.float32)
        target_vector /= np.linalg.norm(target_vector)
        Ry_new = BVHMotion.rotation_matrix_from_vectors(np.array([0, 0, 1], np.float32), target_vector)

        R_raw = res.joint_rotation[frame_num, 0, :]  # 第frame_num帧的朝向
        Ry, Rxz = self.decompose_rotation_with_yaxis(R_raw)

        Ry_new = Ry_new * R.from_quat(Ry).inv()  # 围绕y轴所需的旋转

        for i in range(res.joint_rotation.shape[0]):
            res.joint_rotation[i, 0, :] = (Ry_new * R.from_quat(res.joint_rotation[i, 0, :])).as_quat()

        frame_num = (res.joint_position.shape[0] + frame_num) % res.joint_position.shape[0]
        for idx in range(res.joint_position.shape[0]):
            if idx == frame_num:
                continue
            res.joint_position[idx, 0, :] = res.joint_position[idx, 0, :] - res.joint_position[frame_num, 0, :]
            res.joint_position[idx, 0, :] = (Ry_new).apply(res.joint_position[idx, 0, :])
            res.joint_position[idx, 0, :] += res.joint_position[frame_num, 0, :]

        return res

    @staticmethod
    def rotation_matrix_from_vectors(vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        if np.linalg.norm(vec1 - vec2) < 0.0001:
            return R.from_matrix(np.eye(3))

        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return R.from_matrix(rotation_matrix)


def slerp(v1, v2, t):
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    if np.linalg.norm(v1 - v2) < 0.0001:
        return v1
    # 解决插值绕远路的问题：https://blog.csdn.net/weixin_46477226/article/details/121258542
    if np.dot(v1,v2)<0:
        v1=-v1
    alpha = np.arccos(np.dot(v1, v2))
    result = np.sin((1 - t) * alpha) / np.sin(alpha) * v1 + np.sin(t * alpha) / np.sin(alpha) * v2
    result /= np.linalg.norm(result)
    return result


# part2
def blend_two_motions(bvh_motion1, bvh_motion2, alpha):
    '''
    blend两个bvh动作
    假设两个动作的帧数分别为n1, n2
    alpha: 0~1之间的浮点数组，形状为(n3,)
    返回的动作应该有n3帧，第i帧由(1-alpha[i]) * bvh_motion1[j] + alpha[i] * bvh_motion2[k]得到
    i均匀地遍历0~n3-1的同时，j和k应该均匀地遍历0~n1-1和0~n2-1
    '''

    res = bvh_motion1.raw_copy()
    res.joint_position = np.zeros((len(alpha), res.joint_position.shape[1], res.joint_position.shape[2]))
    res.joint_rotation = np.zeros((len(alpha), res.joint_rotation.shape[1], res.joint_rotation.shape[2]))
    res.joint_rotation[..., 3] = 1.0

    # TODO: 你的代码
    n1 = bvh_motion1.motion_length
    n2 = bvh_motion2.motion_length
    n3 = alpha.shape[0]
    joint_num = res.joint_position.shape[1]
    for i in range(n3):
        j = np.round((i / (n3 - 1)) * (n1 - 1)).astype(np.int32)
        k = np.round((i / (n3 - 1)) * (n2 - 1)).astype(np.int32)
        res.joint_position[i, :, :] = (1 - alpha[i]) * bvh_motion1.joint_position[j, :, :] + alpha[
            i] * bvh_motion2.joint_position[k, :, :]
    for i in range(n3):
        j = np.round((i / (n3 - 1)) * (n1 - 1)).astype(np.int32)
        k = np.round((i / (n3 - 1)) * (n2 - 1)).astype(np.int32)
        for joint in range(joint_num):
            res.joint_rotation[i, joint, :] = slerp(bvh_motion1.joint_rotation[j, joint, :],
                                                    bvh_motion2.joint_rotation[k, joint, :], alpha[i])
    return res


# part3
def build_loop_motion(bvh_motion):
    '''
    将bvh动作变为循环动作
    由于比较复杂,作为福利,不用自己实现
    (当然你也可以自己实现试一下)
    推荐阅读 https://theorangeduck.com/
    Creating Looping Animations from Motion Capture
    '''
    res = bvh_motion.raw_copy()

    from smooth_utils import build_loop_motion
    return build_loop_motion(res)


# part4
def concatenate_two_motions(bvh_motion1, bvh_motion2, mix_frame1, mix_time):
    '''
    将两个bvh动作平滑地连接起来，mix_time表示用于混合的帧数
    混合开始时间是第一个动作的第mix_frame1帧
    虽然某些混合方法可能不需要mix_time，但是为了保证接口一致，我们还是保留这个参数
    Tips:
        你可能需要用到BVHMotion.sub_sequence 和 BVHMotion.append
    '''
    res = bvh_motion1.raw_copy()

    # TODO: 你的代码
    pos = bvh_motion1.joint_position[-1, 0, [0, 2]]
    rot = bvh_motion1.joint_rotation[-1, 0]
    facing_axis = R.from_quat(rot).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
    new_motion2 = bvh_motion2.translation_and_rotation(0, pos, facing_axis)
    #l2 = new_motion2.motion_length
    #l1 = res.motion_length
    #print(l2,l1)
    # 下面这种直接拼肯定是不行的(
    joint_num = res.joint_rotation.shape[1]
    for i in range(mix_time):
        t = i/mix_time
        #k=np.round(i*l2/l1).astype(np.uint32)
        new_motion2.joint_position[i] = t*new_motion2.joint_position[i] +(1-t)*res.joint_position[-mix_time+i]
        for joint in range(joint_num):
            new_motion2.joint_rotation[i,joint,:] = slerp(res.joint_rotation[-mix_time+i,joint,:],new_motion2.joint_rotation[i,joint,:],t)
    res.joint_position = np.concatenate([res.joint_position[:-mix_time], new_motion2.joint_position], axis=0)
    res.joint_rotation = np.concatenate([res.joint_rotation[:-mix_time], new_motion2.joint_rotation], axis=0)

    return res
