import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.stats
import environments
import responses
import mdp_solvers
import approximate_value_iteration

""" Runs (context-dependent online) Bayesian IRL via Metropolis-Hastings sampling, and BIRL for a fixed environment.
    Boltzmann-rational responses and partial information! """


# Parameters: mdp = the 2-agent MDP, n_episodes = no. of episodes, sample_size = no. of samples from proposal, true_beta = true inverse temperature beta.
# Returns: per-episode regret of CDO-BIRL and BIRL in a fixed environment.
def test_run(mdp, n_episodes, sample_size):
    # saving per-episode regret
    regret_cdo_birl = []
    regret_fixed_birl = []

    # saving the acceptance rates
    acceptance_rates = [1]
    fixed_acceptance_rates = [1]

    # saving mean beta
    mean_betas = [round(mdp.beta*sum(mdp.R), 1)]
    fixed_mean_betas = [round(mdp.beta * sum(mdp.R), 1)]

    # saving generated trajectories
    fixed_trajectories = []
    trajectories = []

    # choose initial policy uniformly at random
    first_policy = get_random_policy(mdp)

    # perform approximate value iteration to get the approximately optimal value with oracle knowledge
    print("Computing Approximately Optimal Solution")
    optimal_policy_1, optimal_value, optimal_Q = approximate_value_iteration.approx_value_iteration_boltzmann(mdp)

    # saving sampled rewards and betas from online BIRL
    sampled_rewards = []
    sampled_betas = []
    # saving sampled rewards and betas from fixed environment IRL
    fixed_sampled_rewards = []
    fixed_sampled_betas = []

    # get initial values from priors
    initial_reward = prior_simplex_reward(mdp)
    initial_beta = prior_beta() # mdp.beta*sum(mdp.R)  # prior_beta()
    sampled_rewards.append(initial_reward)
    fixed_sampled_rewards.append(initial_reward)
    sampled_betas.append(initial_beta)
    fixed_sampled_betas.append(initial_beta)

    # save record of conditioned MDPs for online BIRL
    cond_MDPs = []
    # save the fixed environment w.r.t. the first policy
    fixed_cond_MDP = environments.ConditionedMDP(mdp, first_policy)

    for t in range(n_episodes):
        if t == 0:
            # for online BIRL
            policy_1 = first_policy
            # for fixed environment BIRL
            fixed_policy_1 = first_policy
        else:
            """ Online BIRL """
            # calculate mean reward from samples generated in the last episode
            mean_reward = sum(sampled_rewards) / len(sampled_rewards)
            mean_beta = sum(sampled_betas) / len(sampled_betas)
            mean_betas.append(mean_beta)
            temp_mdp = copy.deepcopy(mdp)
            temp_mdp.R = mean_reward
            temp_mdp.beta = mean_beta
            # compute approximately optimal commitment strategy w.r.t. mean_reward and mean_beta
            policy_1, temp_V, temp_Q = approximate_value_iteration.approx_value_iteration_boltzmann(temp_mdp)

            """ Fixed Environment BIRL """
            # calculate mean reward from samples generated in the last episode
            fixed_mean_reward = sum(fixed_sampled_rewards) / len(fixed_sampled_rewards)
            fixed_mean_beta = sum(fixed_sampled_betas) / len(fixed_sampled_betas)
            fixed_mean_betas.append(fixed_mean_beta)
            fixed_temp_mdp = copy.deepcopy(mdp)
            fixed_temp_mdp.R = fixed_mean_reward
            fixed_temp_mdp.beta = fixed_mean_beta
            # compute approximately optimal commitment strategy w.r.t. mean_reward and mean_beta
            fixed_policy_1, fixed_temp_V, fixed_temp_Q = approximate_value_iteration.approx_value_iteration_boltzmann(fixed_temp_mdp)
        """ Evaluation of Online BIRL """
        # get Boltzmann response of agent 2 and corresponding joint policy
        policy_2, joint_policy = responses.boltzmann_response(mdp, policy_1)
        # evaluate policy_1
        achieved_V, achieved_Q = mdp_solvers.policy_evaluation_MG(mdp, joint_policy)
        # save per-episode regret
        regret_cdo_birl.append(optimal_value[mdp.start_state] - achieved_V[mdp.start_state])
        # save cond_MDP w.r.t. policy_1
        cond_MDP = environments.ConditionedMDP(mdp, policy_1)
        cond_MDPs.append(cond_MDP)

        """ Evaluation of Fixed Environment BIRL """
        # get Boltzmann response of agent 2 and corresponding joint policy
        fixed_policy_2, fixed_joint_policy = responses.boltzmann_response(mdp, fixed_policy_1)
        # evaluate policy_1
        fixed_achieved_V, fixed_achieved_Q = mdp_solvers.policy_evaluation_MG(mdp, fixed_joint_policy)
        # save per-episode regret
        regret_fixed_birl.append(optimal_value[mdp.start_state] - fixed_achieved_V[mdp.start_state])
        # saving:
        if mdp.start_state == 0:
            np.savetxt('102_random_mdp_birl.txt', [regret_cdo_birl, regret_fixed_birl, acceptance_rates, fixed_acceptance_rates, mean_betas, fixed_mean_betas])
        else:
            np.savetxt('002_maze_maker_birl.txt', [regret_cdo_birl, regret_fixed_birl, acceptance_rates, fixed_acceptance_rates, mean_betas, fixed_mean_betas])
        # create plots
        # plot_per_episode_regret(mdp, regret_cdo_birl, regret_fixed_birl)
        # no need for the sampling in the last episode
        if t == n_episodes - 1:
            break

        """ Generate Trajectories """
        # get RANDOM trajectory length (minimum of 4 steps, used for both fixed environment and online BIRL in the given episode)
        trajectory_length = 3 + np.random.geometric(1-mdp.gamma)
        print("Trajectory Length", trajectory_length)
        # for both cases we assume to observe the same trajectory in episode 0
        if t == 0:
            tra = get_trajectory(mdp, policy_1, policy_2, trajectory_length)
            trajectories.append(tra)
            fixed_trajectories.append(tra)
        else:
            trajectories.append(get_trajectory(mdp, policy_1, policy_2, trajectory_length))
            fixed_trajectories.append(get_trajectory(mdp, first_policy, fixed_policy_2, trajectory_length))

        """ Sample from Posterior """
        # Fixed Environment
        fixed_sampled_rewards, fixed_sampled_betas, fixed_acceptance_rate = sample_from_posterior_fixed_environment(mdp, sample_size, fixed_cond_MDP, fixed_sampled_rewards[-1], fixed_sampled_betas[-1], fixed_trajectories)
        fixed_acceptance_rates.append(fixed_acceptance_rate)
        print("Episode", t, "Fixed Finished")
        # Varying Environment (CDO-BIRL)
        sampled_rewards, sampled_betas, acceptance_rate = sample_from_posterior(mdp, sample_size, cond_MDPs, sampled_rewards[-1], sampled_betas[-1], trajectories)
        acceptance_rates.append(acceptance_rate)
        print("Episode", t, "Varying Finished")
    return np.array(regret_cdo_birl), np.array(regret_fixed_birl)


""" Sampling for Varying Environments """


def sample_from_posterior(mdp, sample_size, cond_MDPs, last_reward, last_beta, trajectories):
    # samples
    sampled_rewards = []
    sampled_betas = []

    old_likelihood, old_scaling_count = compute_likelihood(cond_MDPs, last_reward, last_beta, trajectories)
    n_acceptances = 0
    for k in range(sample_size):
        proposed_reward, proposed_beta, pdf_proposed_given_last, pdf_last_given_proposed = simplex_proposal_distribution(mdp, last_reward, last_beta)
        likelihood, scaling_count = compute_likelihood(cond_MDPs, proposed_reward, proposed_beta, trajectories)
        if old_scaling_count - scaling_count != 0:
            scale = (old_scaling_count - scaling_count) * 1e+20
        else:
            scale = 1
        p = likelihood * pdf_prior_beta(proposed_beta) * pdf_last_given_proposed * scale
        p_old = old_likelihood * pdf_prior_beta(last_beta) * pdf_proposed_given_last
        quotient = p / p_old
        if np.random.uniform(0, 1) < quotient:
            last_reward = proposed_reward
            last_beta = proposed_beta
            old_likelihood = likelihood
            old_scaling_count = scaling_count
            n_acceptances += 1
            print('CDO-BIRL. Accepted Proposal In Step:', k, 'Beta:', last_beta, 'R:', last_reward[np.argmax(mdp.R)], 'Acceptance Rate:', 100 * n_acceptances / (k+1), '%')
        sampled_rewards.append(last_reward)
        sampled_betas.append(last_beta)
    return sampled_rewards, sampled_betas, n_acceptances / sample_size


def compute_likelihood(cond_MDPs, reward, beta, trajectories):
    likelihood = 1
    # to prevent floating point issues, we shift the likelihood by powers of 10
    scaling_count = 0
    # for each episode we have to solve the cond_MDP with reward function = reward
    for t in range(len(trajectories)):
        cond_MDPs[t].R = reward
        # cond_MDPs[t].beta = beta # unnecessary
        # solve conditioned MDP to get the Q-values
        policy_2, V, Q_matrix = mdp_solvers.value_iteration(cond_MDPs[t])
        # get Boltzmann policy
        Q_exponential = np.exp(beta * Q_matrix)
        for s, b in trajectories[t]:
            likelihood *= Q_exponential[s, b] / np.sum(Q_exponential[s, :])
            if likelihood < 1e-100:
                likelihood *= 1e+20
                scaling_count += 1
    return likelihood, scaling_count


""" For Fixed Environment """


def sample_from_posterior_fixed_environment(mdp, sample_size, cond_MDP, last_reward, last_beta, trajectories):
    # samples
    sampled_rewards = []
    sampled_betas = []

    # concatenate trajectories
    trajectory = [item for sublist in trajectories for item in sublist]

    # Metropolis-Hastings sampling
    old_likelihood, old_scaling_count = compute_likelihood_fixed_environment(cond_MDP, last_reward, last_beta, trajectory)
    # acceptance rate
    n_acceptances = 0
    for k in range(sample_size):
        proposed_reward, proposed_beta, pdf_proposed_given_last, pdf_last_given_proposed = simplex_proposal_distribution(mdp, last_reward, last_beta)
        likelihood, scaling_count = compute_likelihood_fixed_environment(cond_MDP, proposed_reward, proposed_beta, trajectory)
        if old_scaling_count - scaling_count != 0:
            scale = (old_scaling_count - scaling_count) * 1e+20
        else:
            scale = 1
        p = likelihood * pdf_prior_beta(proposed_beta) * pdf_last_given_proposed * scale
        p_old = old_likelihood * pdf_prior_beta(last_beta) * pdf_proposed_given_last
        quotient = p / p_old
        if np.random.uniform(0, 1) < quotient:
            last_reward = proposed_reward
            last_beta = proposed_beta
            old_likelihood = likelihood
            old_scaling_count = scaling_count
            n_acceptances += 1
            print('Fixed Env. Accepted Proposal In Step:', k, 'Beta:', last_beta, 'R:', last_reward[np.argmax(mdp.R)], 'Acceptance Rate:', 100 * n_acceptances / (k+1), '%')
        sampled_rewards.append(last_reward)
        sampled_betas.append(last_beta)
    return sampled_rewards, sampled_betas, n_acceptances / sample_size


def compute_likelihood_fixed_environment(cond_MDP, reward, beta, trajectory):
    likelihood = 1
    cond_MDP.R = reward
    # cond_MDP.beta = beta # unnecessary
    policy_2, V, Q_matrix = mdp_solvers.value_iteration(cond_MDP)
    # to prevent floating point issues, we shift the likelihood by powers of 10
    scaling_count = 0
    Q_exponential = np.exp(beta * Q_matrix)
    for s, b in trajectory:
        likelihood *= Q_exponential[s, b] / np.sum(Q_exponential[s, :])
        if likelihood < 1e-100:
            likelihood *= 1e+20
            scaling_count += 1
    return likelihood, scaling_count


""" Proposals and Priors """


# We use a Dirichlet proposal distribution
def proposal_distribution(mdp, last_reward, last_beta):
    """ REWARD """""
    # Dirichlet proposal independent of last_reward
    proposed_reward = np.random.dirichlet(np.ones(mdp.n_states))

    """ BETA """
    # Gamma proposal - optimistically favouring a rational agent 2 (mean last_beta + 1)
    proposed_beta = np.random.gamma(last_beta, 1 + 1 / last_beta)
    proposed_beta = round(proposed_beta, 1)

    # # suppose beta is known
    # proposed_beta = round(mdp.beta * sum(mdp.R), 1)

    """ PDFs """
    # get pdfs: g(proposed | old)
    pdf_proposed_given_last = scipy.stats.gamma(last_beta, 1 + 1 / last_beta).pdf(proposed_beta) + 1e-10
    # get pdfs: g(old | proposed)
    pdf_last_given_proposed = scipy.stats.gamma(proposed_beta, 1 + 1 / last_beta).pdf(last_beta) + 1e-10
    return proposed_reward, proposed_beta, pdf_proposed_given_last, pdf_last_given_proposed


# We use a proposal distribution on the discretised simplex for the reward function
def simplex_proposal_distribution(mdp, last_reward, last_beta):
    """ REWARD """
    # jump size
    n_steps = 100
    step_size = 0.001
    proposed_reward = last_reward.copy()
    for i in range(n_steps):
        minus_direction = np.random.choice(np.where(proposed_reward > 0+1e-5)[0])
        proposed_reward[minus_direction] -= step_size
        plus_direction = np.random.choice(np.where(proposed_reward < 1-step_size)[0])
        proposed_reward[plus_direction] += step_size
    proposed_reward = np.round(proposed_reward, 3)
    if sum(proposed_reward) > 1+1e-5:
        A = sum(proposed_reward)
        print(A, "Imprecise results due to Python")

    """ BETA """
    # Gamma proposal - optimistically favouring a rational agent 2 (mean last_beta + 1)
    proposed_beta = np.random.gamma(last_beta, 1 + 1 / last_beta)
    proposed_beta = round(proposed_beta, 1)

    # # suppose beta is known
    # proposed_beta = round(mdp.beta * sum(mdp.R), 1)

    """ PDFs """
    # get pdfs: g(proposed | old)
    pdf_proposed_given_last = scipy.stats.gamma(last_beta, 1 + 1 / last_beta).pdf(proposed_beta) + 1e-10
    # get pdfs: g(old | proposed)
    pdf_last_given_proposed = scipy.stats.gamma(proposed_beta, 1 + 1 / last_beta).pdf(last_beta) + 1e-10
    return proposed_reward, proposed_beta, pdf_proposed_given_last, pdf_last_given_proposed


# sample from uniform prior on the discretised simplex
def prior_simplex_reward(mdp):
    reward = np.zeros(mdp.n_states)
    step_size = 0.001
    choices = np.random.choice(mdp.n_states, int(1/step_size))
    for i in range(len(choices)):
        reward[choices[i]] += 0.001
    return reward


def prior_beta():
    return np.random.exponential(500)


def pdf_prior_beta(beta):
    return scipy.stats.expon.pdf(beta, scale=500) + 1e-10


def prior_reward(mdp):
    return np.random.dirichlet(np.ones(mdp.n_states))


""" Additional Functions """


def get_trajectory(mdp, policy_1, policy_2, trajectory_length):
    trajectory = []
    cond_MDP = environments.ConditionedMDP(mdp, policy_1)
    for i in range(trajectory_length):
        state_i = cond_MDP.current_state
        action_i = np.random.choice(cond_MDP.n_actions_2, p=policy_2[cond_MDP.current_state, :])
        cond_MDP.current_state = np.random.choice(cond_MDP.n_states, p=cond_MDP.get_transition_probabilities(state_i, action_i))
        trajectory.append([state_i, action_i])
    return trajectory


# get a random deterministic policy
def get_random_policy(mdp):
    policy = np.zeros([mdp.n_states, mdp.n_actions_1])
    for s in range(mdp.n_states):
        policy[s][np.random.choice(mdp.n_actions_1, 1)] = 1
    return policy


# Plot Results
def plot_per_episode_regret(mdp, regret_cdo_birl, regret_fixed_birl):
    plt.style.use('seaborn')
    plt.plot(regret_cdo_birl, label="Bayesian Context-Dependent Online IRL")
    plt.plot(regret_fixed_birl, label="Bayesian IRL in Fixed Environment")
    plt.legend(fontsize=14)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Per-Episode Regret", fontsize=14)
    if mdp.start_state == 0:
        plt.savefig("Bayes_Random_MDP_per_episode_regret.pdf", bbox_inches='tight')
    else:
        plt.savefig("Bayes_Maze_Maker_per_episode_regret.pdf", bbox_inches='tight')
    plt.close()
