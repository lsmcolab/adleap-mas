import numpy as np
from copy import deepcopy


class ProtocolHandler(object):

    def __init__(self):
        pass

    def check_connection(self, devices):
        # valid devices to form the communication network
        connection_pairs = []
        for i in range(len(devices)):
            distances = []
            for j in range(len(devices)):
                distances.append(np.linalg.norm(\
                    np.array(devices[i].position) - np.array(devices[j].position) ))
                # handling the case of i == j
                if distances[-1] == 0:
                    distances[-1] = np.inf
            
            if min(distances) < devices[i].communication.max_distance:
                connection_pairs.append(distances.index(min(distances)))
            else:
                connection_pairs.append(-1)
        
        return connection_pairs


    def check_communication(self, devices, networks=None):
        if networks==None:
            networks = []
            for ag in devices:
                net = [agent.index for agent in devices if agent.index!=ag.index]
                networks.append(net)
        
        can_exchange_information = []
        devices_indexes =  [device.index for device in devices]
        for i in range(len(devices)):
            # checking the protocol constraints
            # 1. broadcast
            if devices[i].communication.protocol == 'broadcast':
                # everyone within the radius and the team can receive
                # this device message
                for index in networks[i]:
                    for receiver in devices:
                        if receiver.index == index and index != devices[i].index:
                            
                            if np.linalg.norm(np.array(devices[i].position) -\
                             np.array(receiver.position)) < devices[i].communication.max_distance:
                                can_exchange_information.append((devices[i].index,index))
                                self.communicate(devices[i],receiver)


            # 2. p2p
            elif devices[i].communication.protocol == 'p2p':
                # the device that is following the protocol constraints
                # can receive the message
                for index in networks[i]:
                    for receiver in devices:
                        if receiver.index == index and index!=devices[i].index and receiver.communication.protocol == 'p2p':
                            if np.linalg.norm(np.array(devices[i].position) -\
                            np.array(receiver.position)) < devices[i].communication.max_distance:
                                can_exchange_information.append((devices[i].index,index))
                                self.communicate(devices[i],receiver)

            else:
                raise NotImplemented
    
    def communicate(self,device1,device2):
        union_agent = deepcopy(device1.memory['agents'])
        union_fire = deepcopy(device1.memory['fire'])
        for pos in device2.memory['fire']:
            if pos not in union_fire:
                union_fire.append(pos)
        
        for pos in device2.memory['agents']:
            if pos not in union_agent:
                union_agent.append(pos)

        device1.memory['agents'] = deepcopy(union_agent)
        device2.memory['agents'] = deepcopy(union_agent)
        device1.memory['fire'] = deepcopy(union_fire)
        device2.memory['fire'] = deepcopy(union_fire)
        print("85 : ",union_agent,union_fire)
        return
