import numpy as np
import copy
import mdp_solvers

""" This is the BIRL code used for Figure 1. We assume that agent A_2 plays Boltzmann-rational policies. """

""" We begin by defining the MDPs corresponding to the three mazes. States are defined according to
    0  1  2  3
    4  5  6  7
    8  9  10 11. Agent A_2 can take actions into all 4 directions: 0 = North, 1 = West, 2 = South, 3 = East. 
    
    At state 3 agent A_2 obtains reward +1 and the game ends. To model this we add a 13-th terminal state. 
     
     For convenience, we model the routes that A_1 blocks not as actions but directly in the single-agent MDP. """


class Mazes():
    def __init__(self, n_states=13, n_actions_1=4, blocked_states=[5, 6, 7]):
        self.n_states = n_states
        self.n_actions_1 = n_actions_1
        self.n_actions_2 = n_actions_1  # only for us to re-use the mdp solvers
        self.blocked_states = blocked_states
        self.P = np.zeros([self.n_states, self.n_actions_1, self.n_states])
        self.R = np.zeros([self.n_states])
        self.gamma = 0.9
        self.start_state = 4  # agent A_2 begins in state 4
        self.current_state = self.start_state

        # state 3 yields reward 1
        self.R[3] = 1

        # moving to and staying in the terminal state
        for a in range(self.n_actions_1):
            self.P[3, a, 12] = 1
            self.P[12, a, 12] = 1

        # transitions ignoring blocked states for the time being
        self.P[0, 0, 0] = 1
        self.P[0, 1, 1] = 1
        self.P[0, 2, 4] = 1
        self.P[0, 3, 0] = 1

        self.P[1, 0, 1] = 1
        self.P[1, 1, 2] = 1
        self.P[1, 2, 5] = 1
        self.P[1, 3, 0] = 1

        self.P[2, 0, 2] = 1
        self.P[2, 1, 3] = 1
        self.P[2, 2, 6] = 1
        self.P[2, 3, 1] = 1

        # from 3 we move to the terminal state 12
        # self.P[3, 0, 3] = 1
        # self.P[3, 1, 3] = 1
        # self.P[3, 2, 7] = 1
        # self.P[3, 3, 2] = 1

        self.P[4, 0, 0] = 1
        self.P[4, 1, 5] = 1
        self.P[4, 2, 8] = 1
        self.P[4, 3, 4] = 1

        self.P[5, 0, 1] = 1
        self.P[5, 1, 6] = 1
        self.P[5, 2, 9] = 1
        self.P[5, 3, 4] = 1

        self.P[6, 0, 2] = 1
        self.P[6, 1, 7] = 1
        self.P[6, 2, 10] = 1
        self.P[6, 3, 5] = 1

        self.P[7, 0, 3] = 1
        self.P[7, 1, 7] = 1
        self.P[7, 2, 11] = 1
        self.P[7, 3, 6] = 1

        self.P[8, 0, 4] = 1
        self.P[8, 1, 9] = 1
        self.P[8, 2, 8] = 1
        self.P[8, 3, 8] = 1

        self.P[9, 0, 5] = 1
        self.P[9, 1, 10] = 1
        self.P[9, 2, 9] = 1
        self.P[9, 3, 8] = 1

        self.P[10, 0, 6] = 1
        self.P[10, 1, 11] = 1
        self.P[10, 2, 10] = 1
        self.P[10, 3, 9] = 1

        self.P[11, 0, 7] = 1
        self.P[11, 1, 11] = 1
        self.P[11, 2, 11] = 1
        self.P[11, 3, 10] = 1

        # blocked states
        for s in range(self.n_states):
            for a in range(self.n_actions_1):
                for next_state in range(self.n_states):
                    if self.P[s, a, next_state] == 1:
                        if next_state in self.blocked_states:
                            self.P[s, a, next_state] = 0
                            self.P[s, a, s] = 1

    # get P(s'|s,a,b)
    def get_transition_probability(self, state, action_1, next_state):
        return self.P[state, action_1, next_state]

    # get vector P(.|s,a,b)
    def get_transition_probabilities(self, state, action_1):
        return self.P[state, action_1, :]

    # get the reward for the current state
    def get_reward(self, state):
        return self.R[state]


# Boltzmann-rational response
def boltzmann_policy(maze, beta):
    policy, V, Q = mdp_solvers.value_iteration(maze)
    Q_exponential = np.exp(beta * Q)
    for s in range(maze.n_states):
        policy[s, :] = Q_exponential[s, :] / sum(Q_exponential[s, :])
    print(policy[0])
    return policy


""" Bayesian IRL Algorithm via MCMC Sampling """


def sample_from_posterior(maze, beta, observed_policy, n_samples):
    samples = []
    # uniform prior
    samples.append(np.random.dirichlet(np.ones(maze.n_states)))
    old_likelihood = compute_likelihood_of_policy(maze, beta, samples[-1], observed_policy)
    for k in range(n_samples):
        proposal = proposal_distribution(samples[-1])
        likelihood = compute_likelihood_of_policy(maze, beta, proposal, observed_policy)
        if np.random.uniform(0, 1) < (likelihood / old_likelihood):
            print(k, "new sample accepted")
            # print(proposal)
            old_likelihood = likelihood
            samples.append(proposal)
    return samples


# Dirichlet Proposal Distribution
def proposal_distribution(mean):
    return np.random.dirichlet(mean + 1)


# When observing pi^2, we have P(pi^2 | r) = prod_s exp(beta*Q(s, pi^2, r)) / sum_b exp(beta*Q(s, b, r))
def compute_likelihood_of_policy(maze, beta, reward_function, observed_policy):
    likelihood = 1
    temp = 0
    new_mdp = copy.deepcopy(maze)
    new_mdp.R = reward_function
    cond_policy, V, Q = mdp_solvers.value_iteration(new_mdp)
    Q_exponential = np.exp(beta * Q)
    for s in range(maze.n_states):
        for a in range(maze.n_actions_1):
            temp += observed_policy[s, a] * Q_exponential[s, a] / sum(Q_exponential[s, :])
        likelihood *= temp
    return likelihood


# Using the data from all three environments:
def sample_from_posterior_for_all_three(maze, beta, observed_policies, n_samples):
    samples = []
    # uniform prior
    samples.append(np.random.dirichlet(np.ones(maze.n_states)))
    old_likelihood = 1
    for i in range(len(observed_policies)):
        old_likelihood *= compute_likelihood_of_policy(maze, beta, samples[-1], observed_policies[i])
    for k in range(n_samples):
        proposal = proposal_distribution(samples[-1])
        likelihood = 1
        for i in range(len(observed_policies)):
            likelihood *= compute_likelihood_of_policy(maze, beta, proposal, observed_policies[i])
        if np.random.uniform(0, 1) < (likelihood / old_likelihood):
            print(k, "new sample accepted")
            # print(proposal)
            old_likelihood = likelihood
            samples.append(proposal)
    return samples


def project_to_c_d(A, c, d):
    min_A = min(A)
    max_A = max(A)
    B = np.zeros(len(A))
    for i in range(len(A)):
        B[i] = ((A[i] - min_A) * (d - c) / (max_A - min_A)) + c
    return B


# We assume that A_2 plays Boltzmann-rational with inverse temperature beta
beta = 10
mean_rewards = []

""" For all three environments separately. """
blocked_cells = [[5, 6, 7], [1, 2, 10], [2, 5, 6]]
for blocked in blocked_cells:
    maze = Mazes(blocked_states=blocked)
    # the policy played by A_2
    policy = boltzmann_policy(maze, beta)
    # get samples from posterior and mean reward function
    reward_samples = sample_from_posterior(maze, beta, policy, 10000)
    mean_reward_function = np.zeros(maze.n_states)
    for k in range(len(reward_samples)):
        mean_reward_function += reward_samples[k]
    mean_reward_function /= len(reward_samples)
    # Round results roughly for easier translation to latex colours
    mean_reward_function *= 100
    mean_reward_function = np.round(mean_reward_function, 2)
    print("Blocked States", blocked)
    print("Rounded", mean_reward_function)
    # add terminal state reward to state 3
    remove_13 = mean_reward_function[0:12]
    remove_13[3] += mean_reward_function[12]
    # get colour scale
    print(project_to_c_d(remove_13, 0, 80))
    mean_rewards.append(mean_reward_function)

print("Mean Reward Function for all three mazes:", mean_rewards)

""" For all three environments together """
observed_policies = []
for blocked in blocked_cells:
    maze = Mazes(blocked_states=blocked)
    # the policy played by A_2
    observed_policies.append(boltzmann_policy(maze, beta))
# get samples from posterior and mean reward function
reward_samples = sample_from_posterior_for_all_three(maze, beta, observed_policies, 10000)
mean_reward_function = np.zeros(maze.n_states)
for k in range(len(reward_samples)):
    mean_reward_function += reward_samples[k]
mean_reward_function /= len(reward_samples)
# Round results roughly for easier translation to latex colours
mean_reward_function *= 100
mean_reward_function = np.round(mean_reward_function, 2)
print("Rounded", mean_reward_function)
# add terminal state reward to state 3
remove_13 = mean_reward_function[0:12]
remove_13[3] += mean_reward_function[12]
# get colour scale
print(project_to_c_d(remove_13, 0, 80))

""" For pairs of environments. """
observed_policies = []
blocked_cells_pair = [[1, 2, 10], [2, 5, 6]]
for blocked in blocked_cells_pair:
    maze = Mazes(blocked_states=blocked)
    # the policy played by A_2
    observed_policies.append(boltzmann_policy(maze, beta))
# get samples from posterior and mean reward function
reward_samples = sample_from_posterior_for_all_three(maze, beta, observed_policies, 10000)
mean_reward_function = np.zeros(maze.n_states)
for k in range(len(reward_samples)):
    mean_reward_function += reward_samples[k]
mean_reward_function /= len(reward_samples)
# Round results roughly for easier translation to latex colours
mean_reward_function *= 100
mean_reward_function = np.round(mean_reward_function, 2)
print("Blocked Pair", blocked_cells_pair)
print("Rounded", mean_reward_function)
# add terminal state reward to state 3
remove_13 = mean_reward_function[0:12]
remove_13[3] += mean_reward_function[12]
# get colour scale
print(project_to_c_d(remove_13, 0, 80))
