from array import array
import numpy as np


# This method returns true if an object is in the viewer vision angle
def angle_of_gradient(obj, viewer, direction):
    xt = (obj[0] - viewer[0])
    yt = (obj[1] - viewer[1])

    x = np.cos(direction) * xt + np.sin(direction) * yt
    y = -np.sin(direction) * xt + np.cos(direction) * yt
    if (y == 0 and x == 0):
        return 0
    return np.arctan2(y, x)


# This method returns the distance between an object and a viewer
def euclidean_distance(obj, viewer):
    return np.sqrt((obj[0] - viewer[0]) ** 2 + (obj[1] - viewer[1]) ** 2)


# This method returns a dict with its values normalised
def normalise(object):
    total_value = 0.0
    # list
    if type(object) is list:
        for value in object:
            total_value += value
        
        normalised_list = []
        for value in object:
            normalised_list.append(value/total_value)
        return normalised_list 
    # dict
    elif type(object) is dict:
        for key in object:
            total_value += object[key]
        
        normalised_dict = {}
        for key in object:
            normalised_dict[key] = object[key]/total_value   
        return normalised_dict        
    # not implemented types 
    else:
        raise NotImplemented     