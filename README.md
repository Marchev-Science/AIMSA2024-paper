# Testing the NEAT Algorithm on a PSPACE-Complete Problem

Angel Marchev, Jr. (orcidID: 0000-0002-5090-3123)  
Dimitar Lyubchev (orcidID: 0009-0006-3970-6272)  
Nikolay Penchev (orcidID: 0009-0000-8955-2474)  

This repository contains the code used to test the **Neuro-Evolution of Augmenting Topologies (NEAT)** algorithm on a PSPACE-complete problem, specifically the Sokoban puzzle. The experiment is based on the study outlined in the paper:

> [Testing the NEAT Algorithm on a PSPACE-Complete Problem](./Testing_the_NEAT_Algorithm_on_a_PSPACE_Complete_Problem.pdf).

## Project Structure

```bash
.
├── LICENSE                            # License for the project
├── README.md                          # This file
├── Testing_the_NEAT_Algorithm_on_a_PSPACE_Complete_Problem.pdf  # Research paper
├── presentation.md                    # Presentation related to the paper
├── neat_experiments/
│   ├── config-feedforward              # NEAT config for feedforward networks
│   ├── config-feedforward-2            # Additional NEAT configurations
│   ├── config-feedforward-3
│   ├── config-feedforward-4
│   ├── neat-sokoban-v01-2.ipynb        # Jupyter notebook for NEAT experiments on Sokoban
│   └── visualize.py                    # Script for visualizing results
├── sokoban_notebooks/
│   ├── base_experiments/
│   │   ├── dqn.ipynb                   # DQN experiment notebook
│   │   ├── ppo_cnn_policy.ipynb        # PPO with CNN experiment notebook
│   │   ├── ppo_defaults.ipynb          # PPO default configuration notebook
│   │   ├── ppo_optimized.ipynb         # Optimized PPO experiment notebook
│   │   └── q_learning.ipynb            # Q-Learning experiment notebook
└── img/                                
