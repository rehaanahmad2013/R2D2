from r2d2.robot_env import RobotEnv
# from r2d2.controllers.oculus_controller import VRPolicy
from r2d2.controllers.xbox_controller import XboxController

from r2d2.trajectory_utils.misc import collect_trajectory

# Make the robot env
env = RobotEnv()
# controller = VRPolicy()
controller = XboxController()

print('Ready')
collect_trajectory(env, controller=controller, save_np="/iris/u/rehaan/testsave.npz", measure_error=False, save_images=False)