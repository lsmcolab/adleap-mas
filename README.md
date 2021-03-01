# AdLeap-MAS: An Open-source Multi-Agent Simulator for Ad-hoc Learning and Planning

## Introduction

*AdLeap-MAS* represents a novel framework focused on the implementation and simulation of Ad-hoc reasoning domains, which considers the approach of collaborative and adversarial contexts focused on ad-hoc environment learning and planning. 
The framework aims to facilitate the running of experiments in the domain and also re-use existing codes across different environments.
In other words, this proposal aims to minimise the implementation cost related to the process that precedes the domain evaluation, which could include the environment design, components settings and, benchmark set definition, while simultaneously improving the robustness of the environment and minimising the errors carried out due to mistakes made in the code adaptation or implementation.
Through the definition of a component-based architecture, *AdLeap-MAS* implements [*Open-AI Gym*](https://github.com/openai/gym/tree/master/gym) package for *Python 3* as the primary tool to define its base components.
Designed to be an open-source framework and a specialised version of the Open-AI Gym simulator, we offer the base classes for implementing new contexts and scenarios of the community's interest.

## Summary

In this README you can find:

* [GET STARTED](#sec-getstarted)
    * [1. Dependencies](#sec-dependencies)
    * [2. Usage](#sec-usage)
    * [3. How to change the components within the framework?](#sec-components)
* [EXAMPLES](#sec-examples)
    * [1. Level-Foraging Environment](#sec-levelforaging)
    * [2. Truco Environment](#sec-truco)
* [DEVELOPMENT INFORMATION](#sec-development)
* [REFERENCES](#sec-references)

<a name="sec-getstarted"></a>
## GET STARTED

<a name="sec-dependencies"></a>
### 1. Dependencies :pencil: 
    
You must install Python 3 and the OpenAI Gym package to run the framework.
You can use [Python 3 website](https://www.python.org/downloads/) and the [OpenAI Gym GitHub](https://github.com/openai/gym) for information about installation **or**, if you are programming at Linux, run the following command lines:

> **For Python 3:**
>
> `sudo apt-get install software-properties-common & sudo add-apt-repository ppa:deadsnakes/ppa & sudo apt-get update & sudo apt-get install python3.8`

> **For OpenAI Gym (Minimal Install):**
>
> pip install gym

> **For OpenAI Gym (Full Install):**
>
> apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig

**NOTE:** make sure that the [NumPy package](https://numpy.org/install/) is also installed for Python 3 before running the framework.

`pip install numpy` **OR** `pip3 install numpy`

### Windows

To execute this framework on Windows OS, you will need to work within the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10).
We recommend installing *Ubuntu 16.04* and *Ubuntu 18.04* as your Linux distribution into the Windows Subsystem.

Moreover, you must install the [VcXserver Windows X Server](https://sourceforge.net/projects/vcxsrv/) to compile the framework correctly.
The VcXserver will enable the simulated environment visualisation, creating the correct display to run your tests.

**NOTE:** once you started your VcXserver (before running the framework), select the following options on the start screen:

<table border="0" cellspacing="0" cellpadding="0" border-collapse="collapse">
 <tr border="0" cellspacing="0" cellpadding="0" border-collapse="collapse">
    <td><img src="imgs/vcxserver1.PNG" alt="drawing" width="400px"/></td>
    <td><img src="imgs/vcxserver2.PNG" alt="drawing" width="400px"/></td>
 </tr>
 <tr border="0" cellspacing="0" cellpadding="0" border-collapse="collapse">
    <td colspan="2" align="center"><img src="imgs/vcxserver3.PNG" alt="drawing" width="400px"/></td>
 </tr>
</table>

------------------------
<a name="sec-usage"></a>
### 2. Usage :muscle:

With all dependencies installed, you have to download this GitHub project and set it on your local workspace.

To start the framework, you only need to choose an environment and run the file `test_[environment_name].py`.

Via the command line, you can use (within the main project directory):

> `python3 test_[environment_name].py`

That's all folks. At this point, you will have the display popping up and the simulation starting with the default components.

### Understanding

*More information will be documented and presented soon.*

------------------------
<a name="sec-components"></a>
### 3. How to change the components within the framework? :fearful:

*More information will be documented and presented soon.*

------------------------
<a name="sec-examples"></a>
## EXAMPLES

<a name="sec-levelforaging"></a>
### 1. Level-Foraging Environment

Initially introduced to evaluate ad hoc teamwork, the Level-based Foraging domain \[[1](#albrecht2015game), [2](#stone2010adhoc)\] represents a problem in which a team of agents must collaborate to accomplish a certain number of tasks in an environment, optimising the time spent in the activity via active collaboration-coordination.
The agents have a certain level (strength) that defines if it is able to collect an item (e.g., a box) of a specific weight.
The boxes are distributed in the environment, and the agents cannot communicate with their teammates.
The following figure illustrates the idea of the problem.

<img src="imgs/level-based-foraging.PNG" alt="drawing" width="500px"/>

As presented, the AdLeap-MAS can implement this problem, while enabling the simulation of (i) a real-time decision (instead of a turn-based approach) and (ii) an online learning and planning of the problem.
The environment implementation delivers only the visible information to the agents, deferring to them the responsibility to reason about the missing data and build the corresponding belief state.
Additionally, in this domain, the agents have four parameters: level, vision radius, vision angle and type; and the tasks have only one parameter: weight.
The initial position and these parameters are all concealed from the agents.


<a name="sec-truco"></a>
### 2. Truco Environment

A popular card game in Brazil, Truco is played by pairs of people, compounding two teams.
The game starts with dealing three cards for each player and turning up one card on the table.
Each card has a strength associated with its rank and suit, which will compare the cards, one against the other. 
The team's goal is to score 12 points over a maximum of 23 rounds, playing over a best-of-three game system.
The team scores 1 point if they win the best-of-three round.
The following figure illustrates Truco's table for four players playing the game.

<img src="imgs/truco.PNG" alt="drawing" width="500px"/>

Categorising a completely distinct environment from the Level-based Foraging domain, the implementation of this card game has a principal objective to show the versatility offered by the AdLeap-MAS.
Furthermore, the implementation enables the simulation of (i) a turn-based approach for the decision-making process and (ii) an online learning of the problem by receiving partial information mainly via the observation of the adversaries' and teammate's play.
Additionally, the environment delivers to the agents only the visible information, allowing them to reason about the missing data and build their belief state.
Note that even though all the hands are visible in the interface, it is not related to the actual information.

------------------------
<a name="sec-development"></a>
## DEVELOPMENT INFORMATION
**Status:** In development. :computer:

*More information will be documented and presented soon.*

------------------------
<a name="sec-references"></a>
## REFERENCES

<a name="albrecht2015game">[1]</a> Stefano V Albrecht and Subramanian Ramamoorthy. 2015.  A game-theoretic model and best-response learning method for ad hoc coordination in multi agent systems. *arXiv preprint arXiv:1506.01170* (2015).

<a name="stone2010adhoc">[2]</a> Peter Stone, Gal A. Kaminka, Sarit Kraus, and Jeffrey S. Rosenschein. 2010. AdHoc Autonomous Agent Teams: Collaboration without Pre-Coordination. *In Proceedings of the Twenty-Fourth Conference on Artificial Intelligence (AAAI)*.
