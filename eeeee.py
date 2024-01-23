roslaunch beginner_tutorials m1.launch 

roslaunch mr_voice voice.launch 

roslaunch open_manipulator_controller open_manipulator_controller.launch usb_port:=/dev/arm

rviz

roslaunch lingao_navigation navigate.launch map_file:=/home/pcms/maps/fambot_map.yaml

roslaunch lingao_bringup robot.launch 

