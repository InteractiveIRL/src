import numpy as np
import environments
import mdp_solvers

""" Get optimal response of agent 2 to the commitment of agent 1. """


def optimal_response(mdp, policy_1):
    policy_2 = np.zeros([mdp.n_states, mdp.n_actions_1])
    cond_MDP = environments.ConditionedMDP(mdp, policy_1)
    policy_2, V, Q = mdp_solvers.value_iteration(cond_MDP)
    joint_policy = np.zeros([mdp.n_states, mdp.n_actions_1, mdp.n_actions_2])
    for s in range(mdp.n_states):
        for a in range(mdp.n_actions_1):
            for b in range(mdp.n_actions_2):
                joint_policy[s, a, b] = policy_1[s, a] * policy_2[s, b]
    return policy_2, joint_policy


""" Get Boltzmann-rational response of agent 2 to the commitment of agent 1. """


def boltzmann_response(mdp, policy_1):
    policy_2 = np.zeros([mdp.n_states, mdp.n_actions_2])
    cond_MDP = environments.ConditionedMDP(mdp, policy_1)
    optimal_policy_2, V, Q = mdp_solvers.value_iteration(cond_MDP)
    Q_exponential = np.exp(mdp.beta * Q)
    for s in range(mdp.n_states):
        policy_2[s, :] = Q_exponential[s, :] / sum(Q_exponential[s, :])
    joint_policy = np.zeros([mdp.n_states, mdp.n_actions_1, mdp.n_actions_2])
    for s in range(mdp.n_states):
        for a in range(mdp.n_actions_1):
            for b in range(mdp.n_actions_2):
                joint_policy[s, a, b] = policy_1[s, a] * policy_2[s, b]
    print("POL 24", policy_2[24, :])
    return policy_2, joint_policy


""" Get epsilon-greedy response of agent 2 to the commitment of agent 1. """


def epsilon_greedy_response(mdp, policy_1, epsilon):
    policy_2 = np.zeros([mdp.n_states, mdp.n_actions_2])
    cond_MDP = environments.ConditionedMDP(mdp, policy_1)
    policy_2, V, Q = mdp_solvers.value_iteration(cond_MDP)
    policy_2[policy_2 == 1] = 1 - epsilon + epsilon / 4
    policy_2[policy_2 == 0] = epsilon / 4
    joint_policy = np.zeros([mdp.n_states, mdp.n_actions_1, mdp.n_actions_2])
    for s in range(mdp.n_states):
        for a in range(mdp.n_actions_1):
            for b in range(mdp.n_actions_2):
                joint_policy[s, a, b] = policy_1[s, a] * policy_2[s, b]
    return policy_2, joint_policy