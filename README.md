# DMCL: Robot Autonomous Navigation via Depth Image Masked Contrastive Learning

## Abstract

​	Achieving high performance in deep reinforcement learning relies heavily on the ability to obtain good state representations from pixel inputs. However, learning an observation-space-to-action-space mapping from high-dimensional inputs is challenging in reinforcement learning, particularly when dealing with consecutive depth images as input states. In addition, we observe that the consecutive inputs of depth images are highly correlated for the autonomous navigation of a mobile robot, which inspires us to capture temporal correlations between consecutive inputs and infer scene change relationships. To this end, we propose a novel end-to-end robot vision navigation method dubbed DMCL, which obtains good spatial-temporal state representation via Depth image Masked Contrastive Learning. It reconstructs the latent representation from consecutive depth images masked in both spatial and temporal dimensions, resulting in a complete environment state representation. To obtain the optimal navigation policy, we leverage the Soft Actor-Critic reinforcement learning in conjunction with the above representation learning. Extensive experiments demonstrate that the proposed DMCL outperforms representative state-of-the-art methods. The source code will be made publicly available.

#### Main dependencies: 

* [ROS Melodic](http://wiki.ros.org/melodic/Installation)
* [PyTorch](https://pytorch.org/get-started/locally/)

#### Clone the repository:

```shell
$ cd ~
### Clone this repo
$ git clone https://github.com/LanrenzzzZ/DMCL
```

#### Compile the workspace:

```shell
$ cd ~/DMCL/catkin_ws
### Compile
$ catkin_make
```

#### Open a terminal and set up sources:

```shell
$ export ROS_HOSTNAME=localhost
$ export ROS_MASTER_URI=http://localhost:11311
$ export ROS_PORT_SIM=11311
$ export GAZEBO_RESOURCE_PATH=~/DMCL/catkin_ws/src/multi_robot_scenario/launch
$ source ~/.bashrc
$ cd ~/DMCL/catkin_ws
$ source devel/setup.bash
$ source install/setup.bash --extend
$ cd ~/DMCL/SAC
$ conda activate DMCL
$ python3 train.py
```

#### To kill the training process:

```shell
$ killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient python python3
```

## Citation

```bibtex
@inproceedings{jiang2023dmcl,
  title={DMCL: Robot Autonomous Navigation via Depth Image Masked Contrastive Learning},
  author={Jiang, Jiahao and Li, Ping and Lv, Xudong and Yang, Yuxiang},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={5172--5178},
  year={2023},
  organization={IEEE}
}
```
