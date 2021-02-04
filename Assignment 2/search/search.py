"""Version 1.2"""

from typing import Tuple, List
from copy import deepcopy
import time

from gym_minigrid.minigrid import MiniGridEnv
import numpy as np

from search import graph as G



def heuristic(state: G.MiniGridState) -> float:
    # TODO: 1
    image = state.obs['image'][:, :, 0]
    direction = state.obs['direction']

    goal_pos = np.where(image == 8)
    Goal_X = goal_pos[0]
    Goal_Y = goal_pos[1]
    Agent_X = state.env.agent_pos[0]
    Agent_Y = state.env.agent_pos[1]

    # Manhattan Distance
    dx = abs(Agent_X - Goal_X)
    dy = abs(Agent_Y - Goal_Y)
    return dx + dy


def search(init_state: G.MiniGridState, frontier: G.DataStructure) -> Tuple[List[int], int]:
    # TODO: 2
    root = G.SearchTreeNode(init_state, None, -1,  0)
    # initialize the frontier
    frontier.add(root, heuristic(root.state) + 0)
    # initialize the explored set
    explored_nodes = []
    num_explored_nodes = 0
    # initialize the plan
    plan = []

    while(1):
        if frontier.is_empty():
            break
        else: 
            # choose a leaf node and remove it from the frontier
            cur_node = frontier.remove()

            # if the node contains a goal state
            if cur_node.state.is_goal():
                node_plan = cur_node.get_path()
                for i in node_plan:
                    if i.action != -1:
                        plan.append(i.action)
                break
            else: 
                # add the node to the explored set
                explored_nodes.append(cur_node.state)
                num_explored_nodes += 1

                for action in range(3):
                    if action == 2:
                        # path cost of action 2 is 1
                        Path_Cost = 1
                    else:
                        # path cost of action 0 and 1 is 2
                        Path_Cost = 2

                    # if the node is not in the frontier or explored set
                    if (not frontier.is_in(cur_node.state.successor(action))) and (cur_node.state.successor(action) not in explored_nodes):
                        # expand the chosen node
                        new_state = G.SearchTreeNode(cur_node.state.successor(action), cur_node, action, cur_node.path_cost + Path_Cost)
                        # add the resulting node to the frontier                
                        frontier.add(new_state, heuristic(new_state.state) + new_state.path_cost)

    return plan, num_explored_nodes

def execute(init_state: G.MiniGridState, plan: List[int], delay=0.5) -> float:
    env = deepcopy(init_state.env)
    env.render()
    sum_reward = 0
    for i, action in enumerate(plan):
        print(f'action no: {i} = {action}')
        time.sleep(delay)
        _obs, reward, done, _info = env.step(action)
        sum_reward += reward
        env.render()
        if done:
            break
    env.close()
    return sum_reward
