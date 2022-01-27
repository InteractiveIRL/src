import numpy as np

""" 7x7 grid 
    State Space:
    0  1  2  3  4  5  6
    7  8  9  .  .  .
    .  .  . 
    .
    . 
    42  43  45  46  47  48
    
    3 states with collectible rewards, so 7x7x2^3 = 392 states in total
        
    Actions:
    Agent 1: 0 = NE, 1 = NS, 2 = NW, 3 = ES, 4 = EW, 5 = SW.
    Agent 2: 0 = North, 1 = East, 2 = South, 3 = West. 
    
"""


class MazeMaker:
    def __init__(self, grid_size=7, n_actions_1=6, n_actions_2=4, reward_states=[36, 13, 47], beta=10):
        self.grid_size = grid_size
        self.n_states = grid_size ** 2 * 8
        self.n_actions_1 = n_actions_1
        self.n_actions_2 = n_actions_2
        self.P = np.zeros([self.n_states, self.n_actions_1, self.n_actions_2, self.n_states])
        self.R = np.zeros([self.n_states])  # reward is a function of state only
        self.gamma = 0.9  # 0.9 for 7x7 - episode ends w.p. 0.1 each time step
        self.certainty = 0.8  # w.p. 0.2 random move to neighbouring cell
        self.start_state = 24  # 24  # np.random.choice(range(0, 48)) # center of the 7x7 grid
        self.current_state = self.start_state
        self.reward_states = reward_states
        self.beta = beta

        no_cells = self.grid_size ** 2
        # set transition probabilities
        for a in range(self.n_actions_1):
            for b in range(self.n_actions_2):
                if a in [0, 1, 2] and b == 0:  # going north
                    for s in range(no_cells):
                        next_state = s - self.grid_size
                        if next_state < 0:  # out of bounds (going north in states 0, 1, 2, ..., 6)
                            self.P[s, a, b, s] = self.certainty  # stay in state
                        else:  # in bounds, going north
                            self.P[s, a, b, next_state] = self.certainty
                if a in [0, 3, 4] and b == 1:  # going east
                    for s in range(no_cells):
                        next_state = s + 1
                        if next_state % self.grid_size == 0:  # out of bounds (going east in states 6, 13, 20, ..., 48)
                            self.P[s, a, b, s] = self.certainty  # stay
                        else:
                            self.P[s, a, b, next_state] = self.certainty
                if a in [1, 3, 5] and b == 2:  # going south
                    for s in range(no_cells):
                        next_state = s + self.grid_size
                        if next_state > no_cells - 1:  # out of bounds (going south in states 42, 43, ..., 48)
                            self.P[s, a, b, s] = self.certainty  # stay
                        else:
                            self.P[s, a, b, next_state] = self.certainty
                if a in [2, 4, 5] and b == 3:  # going west
                    for s in range(no_cells):
                        next_state = s - 1
                        if s % self.grid_size == 0:  # out of bounds (going west in states 0, 7, 14, ..., 42)
                            self.P[s, a, b, s] = self.certainty  # stay
                        else:
                            self.P[s, a, b, next_state] = self.certainty
                if (b == 0 and a not in [0, 1, 2]) or (b == 1 and a not in [0, 3, 4]) or (b == 2 and a not in [1, 3, 5]) or (b == 3 and a not in [2, 4, 5]):  # door closed
                    for s in range(no_cells):
                        self.P[s, a, b, s] = self.certainty  # stay
        # set random moves w.p. 1 - self.certainty
        random_move = (1 - self.certainty) / 4
        for s in range(no_cells):
            for a in range(self.n_actions_1):
                for b in range(self.n_actions_2):
                    if s - self.grid_size < 0:  # can't go further up, so the random move upwards means staying in state 0, 1, 2, ..., 6
                        self.P[s, a, b, s] += random_move
                    else:
                        self.P[s, a, b, s - self.grid_size] += random_move  # add 0.05 probability to moving upwards
                    if (s + 1) % self.grid_size == 0:  # can't go further east, so the random move east means staying in state 6, 13, 20, ...
                        self.P[s, a, b, s] += random_move
                    else:
                        self.P[s, a, b, s + 1] += random_move
                    if s + self.grid_size >= no_cells:  # can't go further south, so the random move south mean staying in state 42, 43, ...
                        self.P[s, a, b, s] += random_move
                    else:
                        self.P[s, a, b, s + self.grid_size] += random_move
                    if s % self.grid_size == 0:
                        self.P[s, a, b, s] += random_move
                    else:
                        self.P[s, a, b, s - 1] += random_move

        # copy transitions for other 7 cases
        for k in range(7):
            self.P[(k + 1) * no_cells:(k + 2) * no_cells, :, :, (k + 1) * no_cells:(k + 2) * no_cells] = self.P[0:no_cells, :, :, 0:no_cells]

        # fields with reward +1, +2 and +3
        r_1 = self.reward_states[0]
        r_2 = self.reward_states[1]
        r_3 = self.reward_states[2]
        # all rewards available
        self.R[r_1] = 1
        self.R[r_2] = 2
        self.R[r_3] = 3
        self.P[r_1, :, :, :] = self.P[r_1 + no_cells, :, :, :]  # change to cells where reward 1 has been already collected when entering state r_1
        self.P[r_2, :, :, :] = self.P[r_2 + 2 * no_cells, :, :, :]  # when in state r_2
        self.P[r_3, :, :, :] = self.P[r_3 + 3 * no_cells, :, :, :]  # when in state r_3
        # r_1 not available
        self.R[r_2 + no_cells] = 2
        self.R[r_3 + no_cells] = 3
        self.P[r_2 + no_cells, :, :, :] = self.P[r_2 + 4 * no_cells, :, :, :]  # go to r_1 and r_2 not available
        self.P[r_3 + no_cells, :, :, :] = self.P[r_3 + 5 * no_cells, :, :, :]  # go to r_1 and r_3 not available
        # r_2 not available
        self.R[r_1 + 2 * no_cells] = 1
        self.R[r_3 + 2 * no_cells] = 3
        self.P[r_1 + 2 * no_cells, :, :, :] = self.P[r_1 + 4 * no_cells, :, :, :]  # go to r_1 and r_2 not available
        self.P[r_3 + 2 * no_cells, :, :, :] = self.P[r_3 + 6 * no_cells, :, :, :]  # go to r_2 and r_3 not available
        # r_3 not available
        self.R[r_1 + 3 * no_cells] = 1
        self.R[r_2 + 3 * no_cells] = 2
        self.P[r_1 + 3 * no_cells, :, :, :] = self.P[r_1 + 5 * no_cells, :, :, :]  # go to r_1 and r_3 not available
        self.P[r_2 + 3 * no_cells, :, :, :] = self.P[r_2 + 6 * no_cells, :, :, :]  # go to r_2 and r_3 not available
        # r_1 and r_2 not available
        self.R[r_3 + 4 * no_cells] = 3
        self.P[r_3 + 4 * no_cells, :, :, :] = self.P[r_3 + 7 * no_cells, :, :, :]  # non available
        # r_1 and r_3 not available
        self.R[r_2 + 5 * no_cells] = 2
        self.P[r_2 + 5 * no_cells, :, :, :] = self.P[r_2 + 7 * no_cells, :, :, :]  # non available
        # r_2 and r_3 not available
        self.R[r_1 + 6 * no_cells] = 1
        self.P[r_1 + 6 * no_cells, :, :, :] = self.P[r_1 + 7 * no_cells, :, :, :]  # non available
        # r_1 and r_2 and r_3 not available

    # get a single P(s'|s,a,b)
    def get_transition_probability(self, state, action_1, action_2, next_state):
        return self.P[state, action_1, action_2, next_state]

    # get the vector P(.|s,a,b)
    def get_transition_probabilities(self, state, action_1, action_2):
        return self.P[state, action_1, action_2, :]

    # get the reward for the current state
    def get_reward(self, state):
        return self.R[state]

    def get_marginalised_transition_matrix(self, joint_policy):
        marg_trans = np.zeros([self.n_states, self.n_states])
        for s in range(self.n_states):
            for next_state in range(self.n_states):
                policy_index = np.where(joint_policy[s, :, :] == 1)
                a = policy_index[0]
                b = policy_index[1]
                marg_trans[s, next_state] = self.P[s, a, b, next_state]
        return marg_trans


""" Random MDPs. """


class RandomMDP:
    def __init__(self, n_states, n_actions_1, n_actions_2, beta=10):
        self.n_states = n_states
        self.n_actions_1 = n_actions_1
        self.n_actions_2 = n_actions_2
        self.P = np.zeros([n_states, n_actions_1, n_actions_2, n_states])
        self.R = np.zeros([n_states])  # reward is a function of state only
        self.gamma = 0.9
        self.start_state = 0
        self.current_state = self.start_state
        self.beta = beta

        for s in range(self.n_states):
            for a in range(self.n_actions_1):
                for b in range(self.n_actions_2):
                    self.P[s, a, b] = np.random.dirichlet(np.ones(self.n_states) / self.n_states)
        # round the probabilities to avoid floating point errors
        self.P = np.round(self.P, 5)
        for s in range(self.n_states):
            for a in range(self.n_actions_1):
                for b in range(self.n_actions_2):
                    summed = 1 - sum(self.P[s, a, b, :])
                    if summed != 0:
                        indices = np.where((0 < self.P[s, a, b, :] + summed) & (self.P[s, a, b, :] + summed < 1))[0]
                        while True:
                            index = np.random.choice(indices)
                            if 0 < self.P[s, a, b, index] + summed < 1:
                                self.P[s, a, b, index] += summed
                                break
                            else:
                                print("ERROR when defining transition probabilities")

        # uniform initial state distribution encoded in state 0
        for a in range(self.n_actions_1):
            for b in range(self.n_actions_2):
                self.P[0, a, b] = np.ones(self.n_states) / self.n_states
        # setting reward function - param 0.5, 0.5 or 0.5, 0.7
        self.R = np.round(np.random.beta(0.5, 0.7, self.n_states), 2)

    # get a single P(s'|s,a,b)
    def get_transition_probability(self, state, action_1, action_2, next_state):
        return self.P[state, action_1, action_2, next_state]

    # get the vector P(.|s,a,b)
    def get_transition_probabilities(self, state, action_1, action_2):
        return self.P[state, action_1, action_2, :]

    # get the reward for the current state
    def get_reward(self, state):
        return self.R[state]

    def get_marginalised_transition_matrix(self, joint_policy):
        marg_trans = np.zeros([self.n_states, self.n_states])
        for s in range(self.n_states):
            for next_state in range(self.n_states):
                policy_index = np.where(joint_policy[s, :, :] == 1)
                a = policy_index[0]
                b = policy_index[1]
                marg_trans[s, next_state] = self.P[s, a, b, next_state]
        return marg_trans


# Define the marginalized MDP given a policy from agent 1
class ConditionedMDP:
    def __init__(self, mdp, policy):
        self.n_states = mdp.n_states
        self.n_actions_1 = mdp.n_actions_1
        self.n_actions_2 = mdp.n_actions_2
        self.P = np.zeros([self.n_states, self.n_actions_2, self.n_states])
        self.R = mdp.R  # reward is a function of state only
        self.gamma = mdp.gamma
        self.policy = policy  # policy of agent 1
        self.start_state = mdp.start_state
        self.current_state = self.start_state
        self.beta = mdp.beta

        for s in range(self.n_states):
            for b in range(self.n_actions_2):
                for next_state in range(self.n_states):
                    self.P[s, b, next_state] = np.dot(mdp.P[s, :, b, next_state], self.policy[s, :])

    # get a single P(j|s,a)
    def get_transition_probability(self, state, action, next_state):
        return self.P[state, action, next_state]

    # get the vector P( . | s,a)
    def get_transition_probabilities(self, state, action):
        return self.P[state, action, :]

    # get the reward for the current state action
    def get_reward(self, state):
        return self.R[state]

    # get the marginalised transition kernel given policies pi_1 and pi_2 (the MDP is already conditioned on policy_1)
    def get_marginalised_transition_kernel(self, policy_2):
        marg_P = np.zeros([self.n_states, self.n_states])
        for s in range(self.n_states):
            for next_state in range(self.n_states):
                marg_P[s, next_state] = np.dot(self.P[s, :, next_state], policy_2[s, :])
        return marg_P
