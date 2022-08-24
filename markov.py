import collections
import itertools
import random
import sys


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


with open(sys.argv[1]) as f:
    corpus = f.read()

size = 100


transitions = collections.defaultdict(lambda: collections.defaultdict(int))

words = corpus.lower().split()

for w, n in pairwise(words):
    transitions[w][n] += 1

sample = []

def total_rand(transitions):
    return random.choice(list(transitions.keys()))

def markov_rand(transitions, current):
    w = transitions[current]
    if not w:
        return total_rand(transitions)
    return random.choices(list(w.keys()), list(w.values()))[0]

for i in range(size):
    if not sample:
        sample.append(total_rand(transitions))
    sample.append(markov_rand(transitions, sample[-1]))

print(' '.join(sample))
