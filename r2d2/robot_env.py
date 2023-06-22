from copy import deepcopy

import gym
import numpy as np

from r2d2.calibration.calibration_utils import load_calibration_info
from r2d2.camera_utils.info import camera_type_dict
# from r2d2.camera_utils.wrappers.multi_camera_wrapper import MultiCameraWrapper
from r2d2.misc.parameters import hand_camera_id, nuc_ip
from r2d2.misc.server_interface import ServerInterface
from r2d2.misc.time import time_ms
from r2d2.misc.transformations import change_pose_frame


class RobotEnv(gym.Env):
    def __init__(self, action_type="cartesian_velocity", camera_kwargs={}):
        # Initialize Gym Environment
        super().__init__()

        # Define Action Space #
        assert action_type in ['cartesian_position', 'joint_position', 'cartesian_velocity', 'joint_velocity']
        self.action_type = action_type
        self.check_action_range = 'velocity' in action_type

        # Robot Configuration
        self.reset_joints = np.array([0, -1/5 * np.pi, 0, -4/5 * np.pi,  0, 3/5 * np.pi, 0.])
        self.randomize_low = np.array([-0.1, -0.2, -0.1, -0.3, -0.3, -0.3])
        self.randomize_high = np.array([0.1, 0.2, 0.1, 0.3, 0.3, 0.3])
        self.DoF = 7 if ('cartesian' in action_type) else 8
        self.control_hz = 10

        self.lower_bounds = [0.27, -0.15, 0.17, -4, -0.5, -0.3]
        self.upper_bounds = [0.55, 0.15, 0.45, 4, 0.5, 0.3]
        
        self.action_space = Box(
            np.array([-1] * (self.DoF)), # dx_low, dy_low, dz_low, dgripper_low
            np.array([ 1] * (self.DoF)), # dx_high, dy_high, dz_high, dgripper_high
        )
        env_obs_spaces = {
            # 'pixels': Box(0, 255, (100, 100, 3), np.uint8),
            'state': Box(
                np.array([-10, -10, -10, -10, -10, -10, -10]),
                np.array([10, 10, 10, 10, 10, 10, 10]),
            ),
        }
        self.observation_space = Dict(env_obs_spaces)
        self.observation_space_dict = env_obs_spaces
        
        if nuc_ip is None:
            from franka.robot import FrankaRobot

            self._robot = FrankaRobot()
        else:
            self._robot = ServerInterface(ip_address=nuc_ip)
            # use if ssh tunneling
            # self._robot = ServerInterface(ip_address='127.0.0.1')

        # Create Cameras
        self.camera_reader = MultiCameraWrapper(camera_kwargs)
        self.calibration_dict = load_calibration_info()
        self.camera_type_dict = camera_type_dict

        self.curstep = 0
        # Reset Robot
        self.reset()

    def cv2_reward(self, observation):
        image = observation
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)                                      # 0 - Set color space
        lower_orange = np.array([10, 75, 170])
        upper_orange = np.array([30, 194, 257])
        mask_image = cv2.inRange(hsv, lower_orange, upper_orange)                             # 1 - Mask only blue
        result_image = cv2.bitwise_and(image,image, mask= mask_image)                     # 2 - Convert masked color to white. Rest to black 
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY )                    # 3 - Convert to Gray (needed for binarizing)

        contours, _ = cv2.findContours(result_image,
                                    cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)                           # 4 - Find contours of all shapes
        contours = sorted(contours, key=cv2.contourArea, reverse=True) [:1]               # 5 - Select biggest; drop the rest
        if len (contours) > 0:
            cnt = contours[0]
            x,y,w,h = cv2.boundingRect(cnt)
            centroid_x = x + (w/2)
            centroid_y = y + (h/2)

            if centroid_y < 40:
                return 1
        return 0
        
    def reinit_robot(self):
        self._robot = ServerInterface(ip_address=nuc_ip)

    def step(self, action):
        # Check Action
        assert len(action) == self.DoF
        if self.check_action_range:
            assert (action.max() <= 1) and (action.min() >= -1)

        s_dict, _ = self._robot.get_robot_state()
        cartpos = s_dict['cartesian_position']
        for i in range(0, 6):
            if cartpos[i] < self.lower_bounds[i]:
                action[i] = max(0, action[i])
            elif cartpos[i] > self.upper_bounds[i]:
                action[i] = min(0, action[i])
                
        # Update Robot
        action_info = self.update_robot(action, action_space=self.action_type)
        self.curstep += 1

        newdict = {}
        frankaobs = np.asarray(action_info['cartesian_position'], dtype=np.float32)
        newdict['state'] = np.concatenate([frankaobs, np.asarray([self.get_gripper_pos()])])
        newdict['pixels'] = self.get_images()
        return newdict, self.cv2_reward(newdict['pixels']), (self.curstep >= 150), {}

    def reset(self, randomize=False):
        self._robot.update_gripper(0, velocity=False, blocking=True)

        if randomize:
            noise = np.random.uniform(low=self.randomize_low, high=self.randomize_high)
        else:
            noise = None

        self.curstep = 0
        self._robot.update_joints(self.reset_joints, velocity=False, blocking=True, cartesian_noise=noise)
        state_dict, _ = self._robot.get_robot_state()
        newdict = {}
        frankaobs = np.asarray(state_dict['cartesian_position'], dtype=np.float32)
        newdict['state'] = np.concatenate([frankaobs, np.asarray([self.get_gripper_pos()])])
        newdict['pixels'] = self.get_images()
        return newdict

    def update_robot(self, action, action_space="cartesian_velocity", blocking=False):
        action_info = self._robot.update_command(action, action_space=action_space, blocking=blocking)
        return action_info

    def create_action_dict(self, action):
        return self._robot.create_action_dict(action)

    def get_images(self):
        camera_feed = []
        camera_feed.extend(self._robot.read_cameras())
        return camera_feed[0]['array']

    def read_cameras(self):
        return self.camera_reader.read_cameras()

    def get_gripper_pos(self):
        return self._robot.get_gripper_position()

    def render(self, height=100, width=100, camera_id=None, mode=None):
        image_obs = self.get_images()
        return image_obs

    def get_state(self):
        read_start = time_ms()
        state_dict, timestamp_dict = self._robot.get_robot_state()
        timestamp_dict["read_start"] = read_start
        timestamp_dict["read_end"] = time_ms()
        return state_dict, timestamp_dict

    def get_camera_extrinsics(self, state_dict):
        # Adjust gripper camere by current pose
        extrinsics = deepcopy(self.calibration_dict)
        for cam_id in self.calibration_dict:
            if hand_camera_id not in cam_id:
                continue
            gripper_pose = state_dict["cartesian_position"]
            extrinsics[cam_id + "_gripper_offset"] = extrinsics[cam_id]
            extrinsics[cam_id] = change_pose_frame(extrinsics[cam_id], gripper_pose)
        return extrinsics

    def get_observation(self):
        obs_dict = {"timestamp": {}}

        # Robot State #
        state_dict, timestamp_dict = self.get_state()
        obs_dict["robot_state"] = state_dict
        obs_dict["timestamp"]["robot_state"] = timestamp_dict

        # Camera Readings #
        camera_obs, camera_timestamp = self.read_cameras()
        obs_dict.update(camera_obs)
        obs_dict["timestamp"]["cameras"] = camera_timestamp

        # Camera Info #
        obs_dict["camera_type"] = deepcopy(self.camera_type_dict)
        extrinsics = self.get_camera_extrinsics(state_dict)
        obs_dict["camera_extrinsics"] = extrinsics

        return obs_dict
