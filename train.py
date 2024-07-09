import json
import os
from datetime import datetime
import pandas as pd
from cityflow_env import CityFlowEnvM
from utility import parse_roadnet, plot_data_lists
from a2c_agent import MA2CAgent
import time
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  

def main():
    date = datetime.now().strftime('%Y%m%d_%H%M%S')

    parser=argparse.ArgumentParser()
    parser.add_argument('--d',type=str,default='Syn_1',help='dataset')
    args=parser.parse_args()

    print(str(args.d))

    dataset_path="Datasets/"+str(args.d)+"/data/"

    cityflow_config = {
        "interval": 1,
        "seed": 0,
        "laneChange": False,
        "dir": dataset_path,
        "roadnetFile": "roadnet.json",
        "flowFile": "flow.json",
        "rlTrafficLight": True,
        "saveReplay": False,
        "roadnetLogFile": "replayRoadNet.json",
        "replayLogFile": "replayLogFile.txt"
    }

    with open(os.path.join(dataset_path, "cityflow.config"), "w") as json_file:
        json.dump(cityflow_config, json_file)

    config = {
        'cityflow_config_file': dataset_path+"cityflow.config",
        'epoch': 200,
        'num_step': 3600,  
        'save_freq': 1,
        'phase_step': 10,  
        'model': 'A2C',
    }

    cityflow_config = json.load(open(config['cityflow_config_file']))
    roadnetFile = cityflow_config['dir'] + cityflow_config['roadnetFile']
    config["lane_phase_info"] = parse_roadnet(roadnetFile)

    intersection_id = list(config['lane_phase_info'].keys()) 
    config["intersection_id"] = intersection_id
    phase_list = {id_: config["lane_phase_info"][id_]["phase"] for id_ in intersection_id}
    config["phase_list"] = phase_list

    model_dir = "model/{}_{}".format(config['model'], date)
    result_dir = "result/{}_{}".format(config['model'], date)
    config["result_dir"] = result_dir

    if not os.path.exists("model"):
        os.makedirs("model")
    if not os.path.exists("result"):
        os.makedirs("result")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    env = CityFlowEnvM(config["lane_phase_info"],
                       intersection_id,
                       num_step=config["num_step"],
                       thread_num=16,
                       cityflow_config_file=config["cityflow_config_file"]
                       )

    config["state_size"] = env.state_size
    Magent = MA2CAgent(intersection_id,
                       state_size=config["state_size"],
                       phase_list=config["phase_list"],
                       )

    EPISODES = config['epoch']
    total_step = 0
    episode_rewards = {id_: [] for id_ in intersection_id}
    episode_travel_time = []
    for i in range(EPISODES):
        env.reset()
        state = env.get_state()

        episode_length = 0
        episode_reward = {id_: 0 for id_ in intersection_id} 
        while episode_length < config['num_step']:
            action = Magent.choose_action(state) 
            action_phase = {}
            for id_, a in action.items():
                action_phase[id_] = phase_list[id_][a]

            next_state, reward, done = env.step(action_phase, episode_length)  

            for _ in range(config['phase_step'] - 1):
                next_state, reward_, done = env.step(action_phase, episode_length)
                episode_length += 1
                total_step += 1
                for id_ in intersection_id:
                    reward[id_] += reward_[id_]

            for id_ in intersection_id:
                episode_reward[id_] += reward[id_]

            episode_length += 1
            total_step += 1

            Magent.remember(state, action, next_state, reward, done)
            Magent.train(done)
            state = next_state

        episode_travel_time.append(env.eng.get_average_travel_time())
        for id_ in intersection_id:
            episode_rewards[id_].append(episode_reward[id_])

        print('\n')
        print("Episode:{}, travel time:{}".format(i + 1, env.eng.get_average_travel_time()))

        max_reward_id=max(episode_reward,key=episode_reward.get)
        for id_ in intersection_id:
            Magent.agents[id_].actor.model.set_weights(Magent.agents[max_reward_id].actor.model.get_weights())
            Magent.agents[id_].critic.model.set_weights(Magent.agents[max_reward_id].critic.model.get_weights())

    df = pd.DataFrame(episode_rewards)
    df.to_csv(result_dir + '/rewards.csv', index=False)

    df = pd.DataFrame({"travel time": episode_travel_time})
    df.to_csv(result_dir + '/travel time.csv', index=False)

    plot_data_lists([episode_rewards[id_] for id_ in intersection_id], intersection_id,
                    figure_name=result_dir + '/rewards.pdf')
    plot_data_lists([episode_travel_time], ['travel time'], figure_name=result_dir + '/travel time.pdf')

    Magent.save(model_dir + "/{}-{}".format(config['model'], i + 1))


if __name__ == '__main__':
    main()

