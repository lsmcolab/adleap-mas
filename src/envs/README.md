# Environments

### Index

* [General Overview](#Overview)

* [Ad-Hoc Environment](#ad-hoc-environment)

* [Level Foraging Environment](#level-foraging-environment) 
* [*Truco* Environment](#truco-environment)
* [Creating a new environment](#creating-a-new-environment)

## Overview

The repository consists of 2 sample environments. Both environments derive from the *AdhocReasoningEnv* class. This class serves as the main connection of our environments with OpenAI gym implementations. The graphical interface is created using  <insert proper description here> . The new users can directly go ahead and run the *LevelForagingEnv* or *TrucoEnv* and use it like a standard OpenAI environment, after taking a look at the documentation . In a later section, we discuss the process to create a new environment.     

## Ad-Hoc Environment

This is a base class we propose for all our ad-hoc environments. According to our philosophy, each domain can be abstracted into 4 parts - State set, transition function, observation and components. Here, *component* refers to the different unique parts of the environment.  The examples will clarify this further.  The Ad-Hoc Environment has the following expectations from it's derived classes . We also explain this in the later [section](#creating-a-new-environment).

## Level Foraging Environment

This is a multi agent collaboration environment in which different *agents* try to work together to finish certain jobs called *tasks* . The environment is set in a rectangular grid . So, the *components* of this environment are the agents and the tasks. Each agent has 3 properties - *vision radius*, *vision angle* and *level* . Each task has only 1 property - *level* . The tasks are stationary and the agents have 5 possible actions - Move East (0), Move West (1), Move North (2), Move South (3) and Load (4) (which refers to attempting to complete the task) . A group of agents can finish a task if :-

* They are looking at the task , i.e. they are facing the task , and are in the adjoining cell. 
* The sum of levels of all agents trying to finish a task must be greater than the level of the task.

Now we explain the different essential parts of the environment :-

* State state - <Proper Explanation pending>
* Transition Function - This takes the action as an input and updates the environment appropriately. It should return the next state and any additional info as it's output.  Note that states here are instances of the *LevelForagingEnv*  class.
* Observation Function - This takes a state as an input and outputs the observable form of the state. This is the key function used to intrinsically model partial observability within the environment .
* Action space - Set of possible actions. These actions are explained above.  This has to be instance/derived class of *gym.spaces.Spaces* .

A closer look at the code documentation will give more concrete information regarding the implementation. 

## Truco Environment

## Creating a New Environment

