import random
import heapq
import numpy as np

class SquareGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = []
        self.agents = []
        self.targets = []

    def in_bounds(self, id):
        (x, y) = id
        return ((0 <= x < self.width) and (0 <= y < self.height))
    
    def passable(self, id):
        return ((id not in self.walls) and (id not in self.agents) and (id not in self.targets))
    
    def neighbors(self, id):
        (x, y) = id

        results = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)]
        if (x + y) % 2 == 0: 
            results.reverse() # aesthetics

        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)

        return results

    def cost(self, from_node, to_node):
        (x1, y1) = from_node
        (x2, y2) = to_node
        return (abs(x1 - x2) + abs(y1 - y2))

    def update(self,sim_map,start,goal):
        for y in reversed(range(self.height)):
            for x in range(self.width):
                xy =sim_map[x,y]
                if xy == -1:
                    self.walls.append((x,y)),  # obstacle
                if xy == 1 and (x,y) != start and (x,y) != goal:
                    self.agents.append((x,y))
                if xy == np.inf and (x,y) != goal and (x,y) != start:
                    self.targets.append((x,y))

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]

def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def a_star(sim_map, dim_w, dim_h, start, goal):
    # This case occurs when tasks move

    graph = SquareGrid(dim_w,dim_h)
    graph.update(sim_map,start,goal) # allocating the obstacles

    start = (start[0],start[1])
    goal  = ( goal[0], goal[1])

    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from, cost_so_far = {}, {}
    came_from[start], cost_so_far[start] = None, 0
    
    while not frontier.empty():
        current = frontier.get()
        if current == goal:
            break

        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal,next)
                frontier.put(next, priority)
                came_from[next] = current
    
    path = [goal]

    # In some cases, there may be no possible paths
    try:
        while(path[-1] != start):
            path.append(came_from[path[-1]])
        path.reverse()
    except:
        path = []

    return path

def a_star_planning(sim_map, dim_w, dim_h, action_space, start, goal):
    # 1. Defining the path
    path = a_star(sim_map, dim_w, dim_h, start, goal)

    # 2. Translating the path in actions
    actions = []
    for i in range(0,len(path)-1):
        delta = tuple(map(lambda a, b: a - b, path[i+1], path[i])) 
        if delta == (1,0):
            actions.append(0)
        elif delta == (-1,0):
            actions.append(1)
        elif delta == (0,1):
            actions.append(2)
        elif delta == (0,-1):
            actions.append(3)
        else:
            actions.append(-1)

    if len(actions) > 0:
        return actions[0]
    else:
        return random.sample([0,1,2,3],1)[0]
        #raise IndexError