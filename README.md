# Project 1 - Navigation - Deep Reinforcement Learning Nanodegree

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Setup

#### Download and install deepRL-navigation
clone git https://github.com/clzuend/deepRL-navigation.git
cd deepRL-navigation
pip install .

#### Download the Unity Banana Environment
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
2. Place the file in the project folder. 

3. Depending on your operating system you might have to change the ``BANANA_PATH`` in  `Navigation_DQN.ipynb`. 

### Instructions

Open the `Navigation_DQN.ipynb` workbook to initiate the environment and train the agents.

The ``Agent`` class currently supports three types of agents that can be passed using the ``flavor`` parameter:
- ``plain``: Standard DQN agent.
- ``double``: Double DQN agent.
- ``dueling``: Dueling DQN agent.

```python
agent = Agent(state_size=state_size, action_size=action_size, seed=0, hidden_sizes = [64, 64], flavor='plain')
```

The ``Agent`` class additionally has a ``show_network()`` method to visualize the graph of the network:
````python
agent.qnetwork_local.show_network()
```

All agents use a network with two hidden layers. The number of neurons in each layer can be passed in a list to the ``hidden_sizes`` parameter. 

Additional information can be found in the project report: `Report.pdf`
