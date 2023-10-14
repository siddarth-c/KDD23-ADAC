# ADAC-traffic

## Optimizing Traffic Control with Model-Based Learning: A Pessimistic Approach to Data-Efficient Policy Inference

Traffic signal control is an important problem in urban mobility
with a significant potential for economic and environmental impact.
While there is a growing interest in Reinforcement Learning (RL)
for traffic signal control, the work so far has focussed on learn-
ing through simulations which could lead to inaccuracies due to
simplifying assumptions. Instead, real experience data on traffic is
available and could be exploited at minimal costs. Recent progress
in offline or batch RL has enabled just that. Model-based offline RL
methods, in particular, have been shown to generalize from the
experience data much better than others.
We build a model-based learning framework that infers a Markov
Decision Process (MDP) from a dataset collected using a cyclic
traffic signal control policy that is both commonplace and easy
to gather. The MDP is built with pessimistic costs to manage out-
of-distribution scenarios using an adaptive shaping of rewards
which is shown to provide better regularization compared to the
prior related work in addition to being PAC-optimal. Our model is
evaluated on a complex signalised roundabout and a large multi-
intersection environment, demonstrating that highly performant
traffic control policies can be built in a data-efficient manner.

## Installation
1. Use [gharaffanobuild.yml](https://github.com/siddarth-c/KDD23-ADAC/blob/main/gharaffanobuild.yml) to create a conda environment.
2. Install [sumo library](https://www.eclipse.org/sumo/) for traffic simulation.
3. Unzip [TrafQ.zip](https://github.com/siddarth-c/KDD23-ADAC/blob/main/TrafQ.zip) file in the same directory as _gharaffaEnv.py_.

## Data

Folder 'buffers' provides a small data set collected from cyclic traffic signal control policy.

To generate data sets with different sizes and behavioral policy, check the functionality provided in run-offline-rl.py program.

## Policy building and evaluation

Use the script eval-dac-policies.sh to try out model-based offline RL solutions using the data set provided in folder buffers.


## Cite

```
    @inproceedings{10.1145/3580305.3599459,
    author = {Kunjir, Mayuresh and Chawla, Sanjay and Chandrasekar, Siddarth and Jay, Devika and Ravindran, Balaraman},
    title = {Optimizing Traffic Control with Model-Based Learning: A Pessimistic Approach to Data-Efficient Policy Inference},
    year = {2023},
    isbn = {9798400701030},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3580305.3599459},
    doi = {10.1145/3580305.3599459}}
```
