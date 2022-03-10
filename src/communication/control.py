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


    def check_communication(self, devices, networks):
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
                                self.communicate(devices[i],devices[index])


            # 2. p2p
            elif devices[i].communication.protocol == 'p2p':
                # the device that is following the protocol constraints
                # can receive the message
                for index in networks[i]:
                    for receiver in devices:
                        if receiver.index == index and index!=devices[i].index and receiver.communication.protocol == 'p2p':
                            if np.linalg.norm(np.array(devices[i].position) -\
                            np.array(receiver.position)) < devices[i].communication.max_distance:
                                can_exchange_information.append(devices[i].index,index)
                                self.communicate(devices[i],devices[index])

            else:
                raise NotImplemented
    
    def communicate(self,device1,device2):
        union_agent = list(np.union1d(device1.memory['agent'],device2.memory['agent']))
        union_fire = deepcopy(device1.memory['fire'])
        for pos in device2.memory['fire']:
            if pos not in union_fire:
                union_fire.append(pos)

        device1.memory['agent'] = deepcopy(union_agent)
        device2.memory['agent'] = deepcopy(union_agent)
        device1.memory['fire'] = deepcopy(union_fire)
        device2.memory['fire'] = deepcopy(union_fire)

        return
