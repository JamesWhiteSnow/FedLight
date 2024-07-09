# FedLight
Although Reinforcement Learning (RL) has been successfully applied in traffic control, it suffers from the problems of high average vehicle travel time and slow convergence to optimized solutions. This is because, due to the scalability restriction, most existing RL-based methods focus on the optimization of individual intersections while the impact of their cooperation is neglected. Without taking all the correlated intersections as a whole into account, it is difficult to achieve global optimization goals for complex traffic scenarios. To address this issue, this paper proposes a novel federated reinforcement learning approach named FedLight to enable optimal signal control policy generation for multi-intersection traffic scenarios. Inspired by federated learning, our approach supports knowledge sharing among RL agents, whose models are trained using decentralized traffic data at intersections. Based on such model-level collaborations, both the overall convergence rate and control quality can be significantly improved. Comprehensive experimental results demonstrate that compared with the state-of-the-art techniques, our approach can not only achieve better average vehicle travel time for various multi-intersection configurations, but also converge to optimal solutions much faster.

In this project, we open-source the source code of our FedLight approach. 

On Git Hub, we will introduce how to reproduce the results of our experiments in the paper.

For details of our method, please see our [original paper](https://ieeexplore.ieee.org/abstract/document/9586175) at the 2021 58th ACM/IEEE Design Automation Conference (DAC).

Welcome to cite our paper!

```
@inproceedings{ye2021fedlight,
  title={FedLight: Federated Reinforcement Learning for Autonomous Multi-Intersection Traffic Signal Control},
  author={Ye, Yutong and Zhao, Wupan and Wei, Tongquan and Hu, Shiyan and Chen, Mingsong},
  booktitle={2021 58th ACM/IEEE Design Automation Conference (DAC)},
  year={2021},
  pages={847-852},
  doi={10.1109/DAC18074.2021.9586175}}
}
```

## Requirements
Under the root directory, execute the following conda commands to configure the Python environment.
``conda create --name <new_environment_name> --file requirements.txt``

``conda activate <new_environment_name>``

### Simulator installation
Our experiments are implemented on top of the traffic simulator Cityflow. Detailed installation guide files can be found in https://cityflow-project.github.io/

#### 1. Install cpp dependencies
``sudo apt update && sudo apt install -y build-essential cmake``

#### 2. Clone CityFlow project from github
``git clone https://github.com/cityflow-project/CityFlow.git``

#### 3. Go to CityFlow project’s root directory and run
``pip install .``

#### 4. Wait for installation to complete and CityFlow should be successfully installed
``import cityflow``

``eng = cityflow.Engine``

## Run the code
#### Execute the following command to run the experiment over the specified dataset.
``python train.py -d <dataset_name>``

## Datasets
For the experiments, we used both synthetic and real-world traffic datasets provided by https://traffic-signal-control.github.io/dataset.html.
| Dataset Name | Dataset Type | # of intersections |
| :-----------: | :-----------: | :-----------: |
| Syn_1 | Synthetic | 2×2 |
| Syn_2 | Synthetic | 3×3 |
| Syn_3 | Synthetic | 4×4 |
| Real_1 | Real-world | 3×4 |

