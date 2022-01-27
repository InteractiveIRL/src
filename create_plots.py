import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# input: per-episode regret
def plot_per_episode_regret(regret_list, fixed_regret_list, environment):
    length = 31
    average_regret = np.zeros(length)
    fixed_average_regret = np.zeros(length)
    for j in range(length):
        counter = 0
        for i in range(len(regret_list)):
            padded = np.pad(regret_list[i], (0, length - len(regret_list[i])), 'constant')
            if padded[j] != 0:
                average_regret[j] += regret_list[i][j]
                counter += 1
        if counter != 0:
            average_regret[j] /= counter

    for j in range(length):
        counter = 0
        for i in range(len(fixed_regret_list)):
            padded = np.pad(fixed_regret_list[i], (0, length - len(fixed_regret_list[i])), 'constant')
            if padded[j] != 0:
                fixed_average_regret[j] += fixed_regret_list[i][j]
                counter += 1
        if counter != 0:
            fixed_average_regret[j] /= counter

    # get variation / error margins
    error_margin_below = np.ones(length) * 1
    for t in range(length):
        for i in range(np.shape(regret_list)[0]):
            if t < len(regret_list[i]):
                error_margin_below[t] = min(regret_list[i][t], error_margin_below[t])
    error_margin_above = np.zeros(length)
    for t in range(length):
        for i in range(np.shape(regret_list)[0]):
            if t < len(regret_list[i]):
                error_margin_above[t] = max(regret_list[i][t], error_margin_above[t])

    fixed_error_margin_below = np.ones(length) * 1
    for t in range(length):
        for i in range(np.shape(fixed_regret_list)[0]):
            if t < len(fixed_regret_list[i]):
                fixed_error_margin_below[t] = min(fixed_regret_list[i][t], fixed_error_margin_below[t])
    fixed_error_margin_above = np.zeros(length)
    for t in range(length):
        for i in range(np.shape(fixed_regret_list)[0]):
            if t < len(fixed_regret_list[i]):
                fixed_error_margin_above[t] = max(fixed_regret_list[i][t], fixed_error_margin_above[t])

    # plot
    plt.style.use('seaborn')
    if environment == 'Maze Maker LP' or environment == 'Random MDP LP':
        plt.plot(np.linspace(1, length - 1, num=length - 1), average_regret[0:length - 1], label="Interactive IRL via Linear Programming")  # "Bayesian Context-Dependent Online IRL")
        plt.plot(np.linspace(1, length - 1, num=length - 1), fixed_average_regret[0:length - 1], label="Max-Margin IRL in Fixed Environment")  # "Bayesian IRL in Fixed Environment")
    if environment == 'Maze Maker' or environment == 'Random MDP':
        plt.plot(np.linspace(1, length - 1, num=length - 1), average_regret[0:length - 1], label="Bayesian Interactive IRL")
        plt.plot(np.linspace(1, length - 1, num=length - 1), fixed_average_regret[0:length - 1], label="Bayesian IRL in Fixed Environment")

    plt.fill_between(np.linspace(1, length - 1, num=length - 1), error_margin_below[0:length - 1], error_margin_above[0:length - 1], color='blue', alpha=0.1)
    # plt.fill_between(np.linspace(1, length-1, num=length-1), fixed_error_margin_below[0:length-1], fixed_error_margin_above[0:length-1], color='green', alpha=0.1)
    plt.tick_params(axis="x", labelsize=16.5)
    plt.tick_params(axis="y", labelsize=16.5)
    plt.legend(fontsize=19)
    plt.xlabel("Episode", fontsize=22)
    plt.ylim(bottom=0)
    plt.ylabel("Per-Episode Regret", fontsize=22)
    if environment == 'Maze Maker':
        plt.savefig("Bayes_Maze_Maker_per_episode_regret.pdf", bbox_inches='tight')
    elif environment == 'Random MDP':
        plt.savefig("Bayes_Random_MDP_per_episode_regret.pdf", bbox_inches='tight')
    elif environment == 'Maze Maker LP':
        plt.savefig("LP_Maze_Maker_per_episode_regret.pdf", bbox_inches='tight')
    elif environment == 'Random MDP LP':
        plt.savefig("LP_Random_MDP_per_episode_regret.pdf", bbox_inches='tight')
    else:
        print("ERROR NO ENVIRONMENT SPECIFIED")
    plt.show()


