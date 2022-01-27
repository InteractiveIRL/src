import numpy as np
from numpy import unravel_index


# solve single-agent MDP using value iteration
def value_iteration(mdp, delta=0.0001):
    V = np.zeros(mdp.n_states)
    Q = np.zeros([mdp.n_states, mdp.n_actions_2])
    policy = np.zeros([mdp.n_states, mdp.n_actions_2])
    counter = 0
    while True:
        counter += 1
        V_old = V.copy()
        for s in range(mdp.n_states):
            for a in range(mdp.n_actions_2):
                Q[s, a] = mdp.get_reward(s) + mdp.gamma * np.dot(mdp.get_transition_probabilities(s, a), V[:])
            V[s] = max(Q[s, :])
        if max(np.abs(V_old - V)) < delta:
            for s in range(mdp.n_states):
                policy[s][np.argmax(Q[s, :])] = 1
            # print("VI", counter)
            break
    return policy, V, Q


# perform value iteration to solve the Markov game with centralised control
def value_iteration_MG(markov_game, delta=0.0001):
    V = np.zeros(markov_game.n_states)
    Q = np.zeros([markov_game.n_states, markov_game.n_actions_1, markov_game.n_actions_2])
    policy = np.zeros([markov_game.n_states, markov_game.n_actions_1, markov_game.n_actions_2])
    counter = 0
    while True:
        counter += 1
        V_old = V.copy()
        for s in range(markov_game.n_states):
            for a in range(markov_game.n_actions_1):
                for b in range(markov_game.n_actions_2):
                    Q[s, a, b] = markov_game.get_reward(s) + markov_game.gamma * np.dot(markov_game.get_transition_probabilities(s, a, b), V[:])
            V[s] = np.amax(Q[s, :, :])
        if max(np.abs(V_old - V)) < delta:
            for s in range(markov_game.n_states):
                # get the actions with maximal Q-value
                max_index = unravel_index(np.argmax(Q[s, :, :]), Q[s, :, :].shape)
                # policy[s, action_1, action_2]
                policy[s, max_index[0], max_index[1]] = 1
            # print("VI-MG", counter)
            break
    return policy, V, Q


# policy evaluation for joint policies
def policy_evaluation_MG(markov_game, joint_policy, delta=0.0001):
    V = np.zeros(markov_game.n_states)
    Q = np.zeros([markov_game.n_states, markov_game.n_actions_1, markov_game.n_actions_2])
    while True:
        V_old = V.copy()
        for s in range(markov_game.n_states):
            for a in range(markov_game.n_actions_1):
                for b in range(markov_game.n_actions_2):
                    Q[s, a, b] = joint_policy[s, a, b] * (markov_game.get_reward(s) + markov_game.gamma * np.dot(markov_game.get_transition_probabilities(s, a, b), V[:]))
            V[s] = np.sum(Q[s, :, :])
        if max(np.abs(V_old - V)) < delta:
            break
    return V, Q


# calculate the value function of two joint policies
def policy_comparison_MG(markov_game, joint_policy_1, joint_policy_2, delta=0.0001):
    V_1 = np.zeros(markov_game.n_states)
    Q_1 = np.zeros([markov_game.n_states, markov_game.n_actions_1, markov_game.n_actions_2])
    V_2 = np.zeros(markov_game.n_states)
    Q_2 = np.zeros([markov_game.n_states, markov_game.n_actions_1, markov_game.n_actions_2])
    while True:
        V_1_old = V_1.copy()
        V_2_old = V_2.copy()
        for s in range(markov_game.n_states):
            for a in range(markov_game.n_actions_1):
                for b in range(markov_game.n_actions_2):
                    Q_1[s, a, b] = joint_policy_1[s, a, b] * (markov_game.get_reward(s) + markov_game.gamma * np.dot(markov_game.get_transition_probabilities(s, a, b), V_1[:]))
                    Q_2[s, a, b] = joint_policy_2[s, a, b] * (markov_game.get_reward(s) + markov_game.gamma * np.dot(markov_game.get_transition_probabilities(s, a, b), V_2[:]))
            V_1[s] = np.sum(Q_1[s, :, :])
            V_2[s] = np.sum(Q_2[s, :, :])
        if max(np.abs(V_1_old - V_1)) < delta and max(np.abs(V_2_old - V_2)) < delta:
            break
    return V_1, V_2


# # calculate optimal joint policies w.r.t. different rewards functions
# def compute_optimal_joint_policies_for_rewards(markov_game, rewards):
#     optimal_joint_policies = []
#     new_MG = copy.deepcopy(markov_game)
#     for i in range(len(rewards)):
#         new_MG.R = rewards[i]
#         joint_policy, V, Q = value_iteration_MG(new_MG)
#         optimal_joint_policies.append(joint_policy)
#     return optimal_joint_policies
