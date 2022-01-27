import numpy as np
import matplotlib.pyplot as plt
import environments
import mdp_solvers
import approximate_value_iteration
import responses
import copy

""" We compare the approximate value iteration algorithms against playing agent 1's part of the optimal joint policy. """

mdp = environments.RandomMDP(n_states=200, n_actions_1=4, n_actions_2=4)
mdp = environments.MazeMaker()
incorrect_mdp = copy.deepcopy(mdp)

""" Boltzmann-Rational Case: """

# Optimal Joint Policy via Centralised Control
opt_joint_policy, V, Q = mdp_solvers.value_iteration_MG(mdp)
opt_joint_policy_1 = np.zeros([mdp.n_states, mdp.n_actions_1])
for s in range(mdp.n_states):
    for a in range(mdp.n_actions_1):
        opt_joint_policy_1[s, a] = sum(opt_joint_policy[s, a, :])

""" We compare for increasing rationality of agent 2, i.e. increasing beta and 1-epsilon. """

# Boltzmann:
joint_pol_value = []
vi_value = []
incorrect_model_value = []
# betas = 51
for b in np.arange(0, 11)*2.5:
    print("Beta is", b)
    # Approximate Value Iteration for Boltzmann Responses
    mdp.beta = b
    vi_policy_1, vi_V, vi_Q = approximate_value_iteration.approx_value_iteration_boltzmann(mdp)
    # Approximate Value Iteration for Boltzmann Responses with Incorrect Model Estimate
    incorrect_mdp.beta = max(0, b-10)
    incorrect_model_policy_1, incorrect_model_V, incorrect_model_Q = approximate_value_iteration.approx_value_iteration_boltzmann(incorrect_mdp)
    # responses w.r.t. optimal-joint and approx. value iteration
    response_opt_joint, response_opt_joint_joint = responses.boltzmann_response(mdp, opt_joint_policy_1)
    response_vi_policy, response_vi_policy_joint = responses.boltzmann_response(mdp, vi_policy_1)
    response_inc_model_policy, response_inc_model_policy_joint = responses.boltzmann_response(mdp, incorrect_model_policy_1)

    opt_V, vi_V = mdp_solvers.policy_comparison_MG(mdp, response_opt_joint_joint, response_vi_policy_joint)
    joint_pol_value.append(opt_V[mdp.start_state])
    vi_value.append(vi_V[mdp.start_state])

    incorrect_model_V, incorrect_model_Q = mdp_solvers.policy_evaluation_MG(mdp, response_inc_model_policy_joint)
    incorrect_model_value.append(incorrect_model_V[mdp.start_state])


# Boltzmann
plt.style.use('seaborn')
plt.figure(0)
plt.plot(np.arange(0, 11)*2.5, vi_value, label="Approximate Value Iteration")
plt.plot(np.arange(0, 11)*2.5, joint_pol_value, label="Commitment of Optimal Joint Policy")
plt.plot(np.arange(0, 11)*2.5, incorrect_model_value, label="Approximate VI with Incorrect Estimate (beta-5)")
plt.legend(fontsize=19)
plt.xlabel("Beta", fontsize=19)
plt.ylabel("Value", fontsize=19)
plt.savefig("Boltzmann_Value_Iteration_Evaluation.pdf", bbox_inches='tight')
# plt.savefig("Boltzmann_Value_Iteration_Evaluation.png", bbox_inches='tight')


# epsilon-greedy
eps_joint_pol_values = []
eps_vi_values = []
steps = 11
for eps in np.linspace(0, 1, steps):
    print("Epsilon is", eps)
    # Approximate Value Iteration for Epsilon-Greedy Responses
    eps_vi_policy_1, eps_vi_V, eps_vi_Q = approximate_value_iteration.approx_value_iteration_eps_greedy(mdp, eps)
    # responses
    eps_response_opt, eps_response_opt_joint = responses.epsilon_greedy_response(mdp, opt_joint_policy_1, eps)
    eps_response_vi, eps_response_vi_joint = responses.epsilon_greedy_response(mdp, eps_vi_policy_1, eps)
    eps_opt_V, eps_vi_V = mdp_solvers.policy_comparison_MG(mdp, eps_response_opt_joint, eps_response_vi_joint)
    eps_joint_pol_values.insert(0, eps_opt_V[mdp.start_state])
    eps_vi_values.insert(0, eps_vi_V[mdp.start_state])

# Plot epsilon-greedy
plt.style.use('seaborn')
plt.figure(1)
plt.plot(np.linspace(0, 1, steps), eps_vi_values, label="Approximate Value Iteration")
plt.plot(np.linspace(0, 1, steps), eps_joint_pol_values, label="Commitment of Optimal Joint Policy")
plt.legend(fontsize=19)
plt.xlabel("1-Epsilon", fontsize=19)
plt.ylabel("Value", fontsize=19)
plt.savefig("Eps_Greedy_Value_Iteration_Evaluation.pdf", bbox_inches='tight')
# plt.savefig("Eps_Greedy_Value_Iteration_Evaluation.png", bbox_inches='tight')