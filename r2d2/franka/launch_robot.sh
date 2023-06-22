source ~/anaconda3/etc/profile.d/conda.sh
conda activate polymetis-local
pkill -9 run_server
pkill -9 franka_panda_cl
launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.robot_ip=172.16.0.56 &
