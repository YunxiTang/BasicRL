import numpy as np

def entropy(x):
    return -x * np.log(x)

x = np.array([[0.2, 0.2, 0.6], # high-entropy
              [0.1, 0.1, 0.7], # mid-entropy
              [0.1, 0.1, 0.8]]) # low-entropy
y = x * entropy(x)
H = np.sum(y, axis=1, keepdims=1)
print(f'Entropy:\n {H}')


