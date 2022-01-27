import numpy as np


# Approximate Value Iteration for Boltzmann Responses
def approx_value_iteration_boltzmann(mdp, delta=0.0001):
    V = np.zeros(mdp.n_states)
    V_hat = np.zeros(mdp.n_states)
    Q_hat = np.zeros([mdp.n_states, mdp.n_actions_1, mdp.n_actions_2])
    counter = 0
    while True:
        counter += 1
        policy_1 = np.zeros([mdp.n_states, mdp.n_actions_1])
        policy_2 = np.zeros([mdp.n_states, mdp.n_actions_1, mdp.n_actions_2])
        temp_Q = np.zeros([mdp.n_states, mdp.n_actions_1])
        V_old = V.copy()
        V_hat_old = V_hat.copy()
        for s in range(mdp.n_states):
            for a in range(mdp.n_actions_1):
                for b in range(mdp.n_actions_2):
                    Q_hat[s, a, b] = mdp.get_reward(s) + mdp.gamma * np.dot(mdp.get_transition_probabilities(s, a, b), V_hat[:])
                    Q_exponential = np.exp(mdp.beta * Q_hat)
                policy_2[s, a, :] = Q_exponential[s, a, :] / sum(Q_exponential[s, a, :])
                temp_Q[s, a] = mdp.get_reward(s)
                for b in range(mdp.n_actions_2):
                    temp_Q[s, a] += mdp.gamma * np.dot(policy_2[s, a, b] * np.round(mdp.get_transition_probabilities(s, a, b), 50), V[:])
        for state in range(mdp.n_states):
            max_action = np.argmax(temp_Q[state, :])
            policy_1[state][max_action] = 1
            V[state] = temp_Q[state, max_action]
            V_hat[state] = max(Q_hat[state, int(np.nonzero(policy_1[state, :])[0]), :])
        if max(np.abs(V_old - V)) < delta and max(np.abs(V_hat_old - V_hat)) < delta:
            print(counter)
            break
        if counter > 100:
            print("Possibly imprecise result")
            break
    return policy_1, V, temp_Q


# Approximate Value Iteration for Epsilon-Greedy Responses
def approx_value_iteration_eps_greedy(mdp, epsilon, delta=0.0001):
    V = np.zeros(mdp.n_states)
    V_hat = np.zeros(mdp.n_states)
    Q_hat = np.zeros([mdp.n_states, mdp.n_actions_1, mdp.n_actions_2])
    # new transition kernel
    trans_kernel = np.zeros([mdp.n_states, mdp.n_actions_1, mdp.n_actions_2, mdp.n_states])
    for s in range(mdp.n_states):
        for a in range(mdp.n_actions_1):
            for b in range(mdp.n_actions_2):
                for b_2 in range(mdp.n_actions_2):
                    trans_kernel[s, a, b, :] += epsilon * mdp.get_transition_probabilities(s, a, b_2) / mdp.n_actions_2
                trans_kernel[s, a, b, :] += (1-epsilon) * mdp.get_transition_probabilities(s, a, b)
    counter = 0
    while True:
        counter += 1
        policy_1 = np.zeros([mdp.n_states, mdp.n_actions_1])
        policy_2 = np.zeros([mdp.n_states, mdp.n_actions_1])
        temp_Q = np.zeros([mdp.n_states, mdp.n_actions_1])
        V_old = V.copy()
        V_hat_old = V_hat.copy()
        for s in range(mdp.n_states):
            for a in range(mdp.n_actions_1):
                for b in range(mdp.n_actions_2):
                    Q_hat[s, a, b] = mdp.get_reward(s) + mdp.gamma * np.dot(mdp.get_transition_probabilities(s, a, b), V_hat[:])
                policy_2[s, a] = int(np.argmax(Q_hat[s, a, :]))
                temp_Q[s, a] = mdp.get_reward(s) + mdp.gamma * np.dot(trans_kernel[s, a, int(policy_2[s, a]), :], V[:])
        for state in range(mdp.n_states):
            max_action = np.argmax(temp_Q[state, :])
            policy_1[state][max_action] = 1
            V[state] = temp_Q[state, max_action]
            V_hat[state] = max(Q_hat[state, int(np.nonzero(policy_1[state, :])[0]), :])
        if max(np.abs(V_old - V)) < delta and max(np.abs(V_hat_old - V_hat)) < delta:
            print(counter)
            break
        if counter > 120:
            print("Possibly imprecise result")
            break
    return policy_1, V, temp_Q

