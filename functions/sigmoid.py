import numpy as np
def sigmoid(x)->int:
    """
    A simple sigmoid function which takes only one number as input.

    >>> sigmoid(1)
    0.7310585786300049
    >>> sigmoid(0)
    0.5
    >>> sigmoid(-1)
    0.2689414213699951
    """
    return 1 / (1 + np.exp(-x))

def multi_sigmoid(x:list[int]):
    """
    A simple sigmoid function which takes a list of numbers as input.

    >>> multi_sigmoid([1,2,3])
    [0.7310585786300049, 0.8807970779778823, 0.9525741268224334]
    """
    for i in x:
        yield sigmoid(x)


