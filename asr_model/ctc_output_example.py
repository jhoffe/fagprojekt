import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from string import ascii_uppercase
from scipy.special import softmax

A = ascii_uppercase + "|" + '-'
T = 10

p = softmax(np.random.rand(len(A), T), axis=0)

plt.figure(figsize=(8, 14))
sns.heatmap(p, annot=True, fmt=".2f", linewidths=.5, cbar=False, yticklabels=A)
plt.tight_layout()
plt.savefig("ctc_output_examples.png")

