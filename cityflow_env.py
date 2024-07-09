import cityflow
import numpy as np


class CityFlowEnvM(object):
    def __init__(self,
                 lane_phase_info,
                 intersection_id,
                 num_step=2000,
                 thread_num=1,
                 cityflow_config_file='example/config_1x2.json'
                 ):
        self.eng = cityflow.Engine(cityflow_config_file, thread_num=thread_num)
        self.num_step = num_step
        self.intersection_id = intersection_id  
        self.state_size = None
        self.lane_phase_info = lane_phase_info  

        self.current_phase = {}
        self.current_phase_time = {}
        self.start_lane = {}
        self.end_lane = {}
        self.phase_list = {}
        self.phase_startLane_mapping = {}
        self.intersection_lane_mapping = {} 

        for id_ in self.intersection_id:
            self.start_lane[id_] = self.lane_phase_info[id_]['start_lane']
            self.end_lane[id_] = self.lane_phase_info[id_]['end_lane']
            self.phase_startLane_mapping[id_] = self.lane_phase_info[id_]["phase_startLane_mapping"]

            self.phase_list[id_] = self.lane_phase_info[id_]["phase"]
            self.current_phase[id_] = self.phase_list[id_][0]
            self.current_phase_time[id_] = 0
        self.get_state() 

    def reset(self):
        self.eng.reset()

    def step(self, action, now_step):
        done = False
        if (now_step == self.num_step):
            done = True
        for id_, a in action.items():
            if self.current_phase[id_] == a:
                self.current_phase_time[id_] += 1
            else:
                self.current_phase[id_] = a
                self.current_phase_time[id_] = 1
            self.eng.set_tl_phase(id_, self.current_phase[id_])  
        self.eng.next_step()
        return self.get_state(), self.get_reward(),done

    def get_state(self):
        state = {id_: self.get_state_(id_) for id_ in self.intersection_id}
        return state

    def get_state_(self, id_):
        state = self.intersection_info(id_)
        start_vehicle_count = [state['start_lane_vehicle_count'][lane] for lane in self.start_lane[id_]]
        end_vehicle_count = [state['end_lane_vehicle_count'][lane] for lane in self.end_lane[id_]]
        pressure = []
        for i in range(len(self.start_lane[id_])):
            pressure.append(start_vehicle_count[i] - end_vehicle_count[i])
        return_state = pressure + [state['current_phase']]
        return self.preprocess_state(return_state)

    def intersection_info(self, id_):
        state = {}
        state['lane_vehicle_count'] = self.eng.get_lane_vehicle_count()
        state['start_lane_vehicle_count'] = {lane: state['lane_vehicle_count'][lane] for lane in self.start_lane[id_]}
        state['end_lane_vehicle_count'] = {lane: state['lane_vehicle_count'][lane] for lane in self.end_lane[id_]}

        state['current_phase'] = self.current_phase[id_]
        state['current_phase_time'] = self.current_phase_time[id_]
        return state

    def preprocess_state(self, state):
        return_state = np.array(state)
        if self.state_size is None:
            self.state_size = len(return_state.flatten())
        return_state = np.reshape(return_state, [1, self.state_size])
        return return_state

    def get_reward(self):
        reward = {id_: self.get_reward_(id_) for id_ in self.intersection_id}
        return reward

    def get_reward_(self, id_):
        '''
        every agent/intersection's reward
        '''
        state = self.intersection_info(id_)
        start_vehicle_count = [state['start_lane_vehicle_count'][lane] for lane in self.start_lane[id_]]
        end_vehicle_count = [state['end_lane_vehicle_count'][lane] for lane in self.end_lane[id_]]
        pressure = sum(start_vehicle_count) - sum(end_vehicle_count)
        reward = -abs(pressure)
        return reward