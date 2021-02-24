# Easy-AI: a context-free Ad-hoc Teamwork framework forreasoning, planning and decision-making evaluation

## Introduction

This repository presents a generic framework which enables easy implementation and test of reasoning methods into the Ad-hoc Teamwork domain.

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

![](imgs/vcxserver1.PNG)
![](imgs/vcxserver2.PNG)
![](imgs/vcxserver3.PNG)

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

<a name="sec-truco"></a>
### 2. Truco Environment

*More information will be documented and presented soon.*

------------------------
<a name="sec-development"></a>
## DEVELOPMENT INFORMATION
**Status:** In development. :computer:

*More information will be documented and presented soon.*

------------------------
<a name="sec-references"></a>
## REFERENCES

*More information will be documented and presented soon.*
