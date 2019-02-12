import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def trial(num_known):
    QUESTIONS = 100
    assert num_known <= QUESTIONS

    num_guesses = QUESTIONS - num_known
    correct_guesses = sum(
        np.random.binomial(n=1, p=0.25, size=num_guesses)
    )
    return correct_guesses



if __name__ =='__main__':
    num_trials = 101
    x = np.arange(101)
    means = []
    stdevs = []
    for xi in x:
        trials = [trial(xi) for n in range(num_trials)]
        mean = np.mean(trials)
        std = np.std(trials)
        means.append(mean)
        stdevs.append(std)

    result = pd.DataFrame({'known':x, 'mean':means, 'stdev':stdevs})
    print(result)

    ax = plt.subplot()
    result.plot.scatter('known', 'mean', ax=ax)
    result.plot.scatter('known', 'stdev', ax=ax)
    plt.show()





