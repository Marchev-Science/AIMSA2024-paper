# Testing the NEAT Algorithm on a PSPACE-Complete Problem

Angel Marchev, Jr. (orcidID: 0000-0002-5090-3123)  
Dimitar Lyubchev (orcidID: 0009-0006-3970-6272)  
Nikolay Penchev (orcidID: 0009-0000-8955-2474)  

This repository contains the code used to test the **Neuro-Evolution of Augmenting Topologies (NEAT)** algorithm on a PSPACE-complete problem, specifically the Sokoban puzzle. The experiment is based on the study outlined in the paper:

> [Testing the NEAT Algorithm on a PSPACE-Complete Problem](./Testing_the_NEAT_Algorithm_on_a_PSPACE_Complete_Problem.pdf).

**Cite this paper**

Marchev, A., Lyubchev, D., Penchev, N. (2025). Testing the NEAT Algorithm on a PSPACE-Complete Problem. In: Koprinkova-Hristova, P., Kasabov, N. (eds) Artificial Intelligence: Methodology, Systems, and Applications. AIMSA 2024. Lecture Notes in Computer Science(), vol 15462. Springer, Cham. https://doi.org/10.1007/978-3-031-81542-3_9

```
@InProceedings{10.1007/978-3-031-81542-3_9,
author="Marchev, Angel
and Lyubchev, Dimitar
and Penchev, Nikolay",
editor="Koprinkova-Hristova, Petia
and Kasabov, Nikola",
title="Testing the NEAT Algorithm on a PSPACE-Complete Problem",
booktitle="Artificial Intelligence: Methodology, Systems, and Applications",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="104--121",
isbn="978-3-031-81542-3"
}
```
![](/img/solution_1.gif) &nbsp; &nbsp; &nbsp;   ![](/img/solution_2.gif)

## Project Structure

```bash
..
├── LICENSE                            # License for the project
├── README.md                          # This file
├── Testing_the_NEAT_Algorithm_on_a_PSPACE_Complete_Problem.pdf  # Research paper
├── presentation.md    
├── sokoban_env.py                      # we need to override in gym-sokoban
├── sokoban_notebooks/
├── neat_experiments/
│   ├── config-feedforward              # NEAT config for feedforward networks [Refer to Paper Section 2.3]
│   ├── config-feedforward-2            
│   ├── config-feedforward-3
│   ├── config-feedforward-4
│   ├── neat-sokoban-v01-2.ipynb        # Jupyter notebook for NEAT experiments on Sokoban [Refer to Paper Section 2.2 and 2.3]
│   └── visualize.py                    # Script for visualizing results
│   base_experiments/
│   │   ├── dqn.ipynb                   # DQN experiment notebook [Refer to Paper Section 2.2]
│   │   ├── ppo_cnn_policy.ipynb        # PPO with CNN experiment notebook [Refer to Paper Section 2.2]
│   │   ├── ppo_defaults.ipynb          # PPO default configuration notebook [Refer to Paper Section 2.2]
│   │   ├── ppo_optimized.ipynb         # Optimized PPO experiment notebook [Refer to Paper Section 2.2]
│   │   ├── q_learning.ipynb            # Q-Learning experiment notebook [Refer to Paper Section 2.2]
└── img/                                # Directory for storing experiment-related images
```                           
## Cloning and Modifying `gym-sokoban`

In order to train the models on the same level consistently, you need to modify the `gym-sokoban` environment. Follow the steps below to clone the `gym-sokoban` repository and copy over the necessary changes.

### Steps to Modify `gym-sokoban`

1. **Clone the `gym-sokoban` Repository**:
   
   First, clone the official `gym-sokoban` repository:
   
   ```bash
   git clone https://github.com/mpSchrader/gym-sokoban.git
    cd gym-sokoban
    pip install -e .
   cp path/to/our/repo/sokoban_env.py path/to/gym-sokoban/gym_sokoban/envs/sokoban_env.py

   ```

## Conda Environment Setup

This project requires Python dependencies to be installed in a controlled environment to ensure compatibility and reproducibility. We use **conda** to manage this environment. Below are the steps to set up the environment using the provided `sokoban-env.yml` file.

### Why Conda?

**Conda** is a powerful package and environment management system. It helps avoid conflicts between different packages and allows us to manage both Python and non-Python dependencies easily. For this project, using conda ensures that all dependencies, including `gym-sokoban` and `stable-baselines3`, are installed correctly and in a compatible way.

### Steps to Set Up the Conda Environment

1. **Install Conda**:
   If you don't have Conda installed, you can install **Miniconda** or **Anaconda**:
   
   - **Miniconda** (lightweight version):
     Download and install from [here](https://docs.conda.io/en/latest/miniconda.html).
   
   - **Anaconda** (full distribution):
     Download and install from [here](https://www.anaconda.com/products/distribution).

2. Then run 
``` conda env create -f sokoban_env.yml```. Probably you'll be asked to run conda init
3. To activate the conda env
```commandline
conda activate sokoban
```
4. Run ```jupyter lab``` and browse the notebooks

