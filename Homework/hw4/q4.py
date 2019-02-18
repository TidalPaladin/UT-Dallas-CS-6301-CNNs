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

    ax = plt.subplot()
    plt.suptitle('Implicit curve versus known questions')
    result = pd.DataFrame({'Known Questions':x, r'$\mu$':means, r'$\sigma$':stdevs})
    result.set_index('Known Questions', inplace=True)
    plt.ylabel('Implicit Curve')
    result.plot(ax=ax, style='.')
    #result.plot(y=r'$\mu$', ax=ax, style='.')
    #result.plot(y=r'$\sigma$', ax=ax, style='.')
    plt.savefig('figures/q4.png')
