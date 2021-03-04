 Environments

### Index

* [General Overview](#overview)

* [Ad-Hoc Environment](#ad-hoc-environment)

* [Level Foraging Environment](#level-foraging-environment) 
* [*Truco* Environment](#truco-environment)
* [Creating a new environment](#creating-a-new-environment)

## Overview

The repository consists of 2 sample environments. Both environments derive from the *AdhocReasoningEnv* class. This class serves as the main connection of our environments with OpenAI gym implementations. The graphical interface is created using  *insert proper description here* . The new users can directly go ahead and run the *test_levelforaging.py* or *test_truco.py* and use it like a standard OpenAI environment. In a later section, we discuss the process to create a new environment.     

## Ad-Hoc Environment

This is a base class we propose for all our ad-hoc reasoning environments. According to our philosophy, each domain can be abstracted into 4 parts - State set, transition function, observation and components. Here, *component* refers to the different unique parts of the environment.  The examples will clarify this further. We also explain this in this [section](#creating-a-new-environment).

## Level Foraging Environment

This is a multi agent collaboration environment in which different *agents* try to work together to finish certain jobs called *tasks* . The environment is set in a rectangular grid . So, the *components* of this environment are the agents and the tasks. Each agent has 3 properties - *vision radius*, *vision angle* and *level* . Each task has only 1 property - *level* . The tasks are stationary and the agents have 5 possible actions - Move East (0), Move West (1), Move North (2), Move South (3) and Load (4) (which refers to attempting to attempting to complete the task) . A group of agents can finish a task if :-

* They are looking at the task , i.e. they are facing the task , and are in the adjoining cell and,
* The sum of levels of all agents trying to finish a task must be greater than the level of the task.

Now we explain the different essential parts of the environment :-

* State set - Is used to model the initial state of the environment. It also includes the ending condition for the environment. 
* Transition Function - This function is named ``` levelforaging_transition``` . This takes the action as an input and updates the environment appropriately. It should return the next state and any additional info as it's output.  Note that states here are 2D numpy arrays. 
* Observation Function - This function is named as ```environment_transition``` This takes a state as an input and outputs the observable form of the state. This is the key function used to intrinsically model partial observability within the environment . This output and input are instances of -*LevelForaginEnv* .
* Action space - Set of possible actions. These actions are explained above.  This has to be instance/derived class of *gym.spaces.Spaces* .
* Reward Function - Returns the reward based on 2 consecutive states. This function is named as ```reward```. We would like to stress the importance of compatibility of reward function with partial observability constraint. Since our reasoning agents can only see some part of the environment (based on vision radius and vision angle), the reward must be based on only that information. For that purpose, each call to the ```levelforaging_transition``` also returns a variable *info* which gives the reward. Users can make use of this method for their custom environments, or just have a central (fully observable) reward scheme, according to their needs. 
A closer look at the code documentation will give more concrete information regarding the implementation. 

## Truco Environment


This is a turn based team collaboration environment in which 4 players are divided into 2 teams of 2 and play against each other. The game uses 40 cards, 10 from each set. So, the *components* in the environment are the players and the cards in the game. The cards have a heirarchy within them, which is 4<5<6<7<Q<J<K<A<2<3 and ties are broken using heirarchy within the different sets ,i.e. Dice < Spade < Hearts < Clubs . The game has maximum of 23 rounds and each round is a best of 3 game. The team that wins 2 games wins the round. The aim is to win 12 such rounds. At the start of each round, a card is selected from the deck and the immediately higher ranked card becomes the trump (for ex, if the card is 7, Q becomes the trump) , which becomes the highest ranked card for the round, rest of the orders remaining the same. Team member are seated in alternating order of teams. Now in each round, each player gets 3 cards. They go around the table and each player puts in one card in their move. The player with the highest ranked card wins and the team wins the game. A game consists of a single move from all 4 players and a round is a best of 3 of games. The winner of the previous game starts the new game. 

Now we explain the different essential parts of the environment :-

* State set - Is used to model the initial state of the environment. It also includes the ending condition for the environment, which here means that a team has 12 points. Initial state consists of dictionary containing "face up cards" , "winning cards" , "Round no" , "points" and "wins" . Each player is derived from *AdhocAgent* class. 
* Transition Function - This function is named ``` truco_transition``` . This takes the action as an input and updates the environment appropriately. It should return the next state and any additional info as it's output.  Note that states here are dictionaries that contain the values mentioned above.  
* Observation Function - This function is named as ```truco_observation``` This takes a state as an input and outputs the observable form of the state. In this scenario, partial observability is important because each player does not know the cards that the other payer has. This output and input are instances of -*LevelForaginEnv* .
* Action space - Set of possible actions. These actions are 0, 1, 2 - which refer to playing the 1st, 2nd and 3rd card respectively. It is has to be instance/derived class of *gym.spaces.Spaces* .
* Reward Function - Returns the reward based on 2 consecutive states. This function is named as ```reward```. 

## Creating a New Environment
The new environment should be a derived class from *AdhocReasoningEnv*. To facilitate a generic approach, we use the followinng classes :- 
* StateSet :- This contains information regarding the state. For ex :- In the *LevelForagingEnv*, it includes the state representation, which is a gym.spaces.Box and initial state and the initial components. These allow to reinitialise the environment using ```env.reset()``` .  
* AdhocAgent :- This is the base class offered for the different parts of the environment which can reason regarding their behaviour. The adhoc agent is characterised by an index and a type, where index should be unique and name of type must match with a reasoning script in src/reasoning - for ex *l1*, *l2* . The AdhocAgent has a ```smart_parameter``` dictionary, used to store information for different planning and reasoning methods.
Now to create the new environment, carry out the following steps :-
1) Create the environment class and call the consturctor of *AdHocReasoningEnv* from it. 
2) Define the components of the class. Make sure that reassoning components derive from *AdHocAgent* class.  
3) Define the transition function . It takes in current state and returns next state and info variable. info can be used in multiple ways, one way is suggested in the reward function for *LevelForaginEnv* above. 
4) Define the reward function. It must take the state and previous state as input and return the reward. Note that the state here will vary according to the need, for example for LevelForaging, it is a numpy array. 
5) Define the observation function. The observation function must take an instance of the *CustomEnv* , and return an instance of the *CustomEnv* class. 
6) (Optional) Create a rendering interface for the environment. We have given two samples, which can be used to understand the procedure. 

To sum up, we have the following structure :- 
```python
def end_condition(state):
  return True
def custom_env_transition(action):
  return next_state,info
def reward(cur_state,next_state):
  return 0
def custom_env_transformation(copied_env):
  # Make changes here
  return env
class CustomEnv(AdHocReasoningEnv):
  # The components must be defined before the environment is initialised, For details :- Refer to truco_test.py or levelforaging_test.py
  def __init__(self,components,*args):
        state_set = StateSet( spaces.Box(5,5), end_condition) #The first argument must be changed accordingly. 
        transition_function = custom_env_transition # Implement this
        action_space = spaces.Discrete(5)  # Define this
        reward_function = reward # Implement this function
        observation_space = custom_env_transformation # Implement this function
        super(LevelForagingEnv, self).__init__(state_set, \
                                    transition_function, action_space, reward_function, \
                                     observation_space, components)
  def render():
    return
# Other details can be seen from the code of both environments.
```
We encourage users to go over our framework documentation and contribute to the project by making more and improved environments. This will help in having a standardised set of baselines and implementations. For the new users, this serves as a transition point from reinforcement learning to multi agent reasoning by providing the OpenAI gym API. 
