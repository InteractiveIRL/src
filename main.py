import environments
import linprog_irl
import bayesian_irl

""" 
    This script tests the Algorithms
        1. Interactive IRL via Linear Programming for Optimal Responses and Full Information, and 
        2. Bayesian Interactive IRL via Metropolis-Hastings Sampling for Boltzmann Responses and Partial Information,
    and compares these to the fixed environment case. 
    
    To execute the script simply run main.py. Below, choose either one of the two environments Maze-Maker or Random MDPs, and 
    either execute linprog_irl.test_run or bayesian_irl.test_run (or both).
    
    While running these scripts output txt-files including the per-episode regret and plots of these preliminary results. 
    The results are going to be saved each completed episode into the same folder and are named "Environment-Name"+"regret"+"CDP or Fixed"+"IRL or BIRL".txt.
    For instance, maze_maker_regret_cdo_irl.txt describes the results of our Algorithm 1 in the optimal responses and full information case. 
    maze_maker_regret_fixed_irl.txt contains the per-episode regret for the case of a fixed environment and the max-margin algorithm.
    Similarly, the files ending with "birl" contain the results for the Boltzmann responses and partial information case. 
    The plots are called, for instance, "LP_Maze_Maker_per_episode_regret_NEW.pdf".
    
    "linprog_irl.test_run(...)" runs fairly quickly and results can be observed almost live. "bayesian_irl.test_run(...)" runs relatively long, depending on 
    the chosen number of proposals for each episode. Clearly, the results become better and more precise the larger the sample size. 
    
    The script "main_approximate_value_iteration.py" empirically evaluates the Approximate Value Iteration Algorithms for Boltzmann Responses and 
    Eps-Greedy Responses, as they are included in the appendix of the paper. 
    In addition, the script "figure_1.py" estimates the reward function (using BIRL) in the maze environment illustrated in Figure 1 of the paper. 
    
    """

maze_maker = environments.MazeMaker()
random_mdp = environments.RandomMDP(n_states=200, n_actions_1=4, n_actions_2=4)

# Number of Episodes to be played
n_episodes = 30

# Run Interactive IRL (Algorithm 1) for optimal responses and full information
regret_cdo_irl, regret_fixed_irl = linprog_irl.test_run(maze_maker, n_episodes)

# No. of proposals per episode for Bayesian IRL (use 25.000)
sample_size = 25000

# Run Bayesian Interactive IRL (Algorithm 4) for boltzmann responses and partial information
regret_cdo_birl, regret_fixed_birl = bayesian_irl.test_run(random_mdp, n_episodes, sample_size)
