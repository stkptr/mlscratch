import random
import itertools
import csv
import operator


def dot(vec1, vec2):
    return sum(map(operator.mul, vec1, vec2))


def gauss(center, temperature):
    return min(max(0, center + random.gauss(center, 1/temperature)), 1)


e = 2.71

relu = lambda x: max(0, x)
sigmoid = lambda x: 1/(e**-x+1)


def forward_layer(layer, inputs, activation):
    new_inputs = []
    for weights in layer[1:]:
        value = activation(dot(inputs, weights) / len(inputs))
        new_inputs.append(value)
    return new_inputs


def forward(network, inputs, activation=sigmoid):
    for layer in network:
        inputs = [layer[0]] + inputs
        inputs = forward_layer(layer, inputs, activation)
    return inputs


def make_network(*args):
    network = [[]]
    for i in args:
        network.append(
            [0.5] + [[0.5 for a in enumerate(network[-1])] for b in range(i)])
    return network[2:]


lmap = lambda a, b: list(map(a, b))


def vary_network(network, temperature):
    return lmap(
        lambda layer:
            [gauss(layer[0], temperature)] +
            lmap(lambda weights:
                lmap(lambda center: gauss(center, temperature),
                    weights),
                layer[1:]),
            network)


def performance_print(networks, generation):
    best = networks[:3]
    values = map(lambda x: f'{x[1]:.04}', best)
    s = ', '.join(values)
    print(f'[{generation}] Top 3 performers distances: {s}')
    worst = networks[-3:]
    values = map(lambda x: f'{x[1]:.04}', worst)
    s = ', '.join(values)
    print(f'[{generation}] Bottom 3 performers distances: {s}')


def load_dataset():
    ins = []
    outs = []
    with open('redwine.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        skip = True
        for r in reader:
            if skip:
                skip = False
                continue
            k = list(map(float, r))
            ins.append([
                k[0] / 100,
                k[1],
                k[2],
                k[3] / 100,
                k[4],
                k[5] / 100,
                k[6] / 1000,
                k[7] / 10,
                k[8] / 10,
                k[9] / 100
            ])
            outs.append([k[11]/10])
    return ins, outs


ds = load_dataset()


population = 100
top_n = 20
dataset_inputs = ds[0]
dataset_expected = ds[1]
generations = 200
architecture = [11, 20, 1]
input_n = architecture[0]
output_n = architecture[-1]

objective = lambda outputs, expected: sum(map(lambda x: abs(x[0] - x[1]), zip(outputs, expected)))

temperature = 0.5

t_at = lambda g: temperature

pop = [vary_network(make_network(*architecture), temperature) for i in range(population)]


for g in range(1, generations):
    # calculate performances
    results = []
    for network in pop:
        guesses = map(lambda x: forward(network, x), dataset_inputs)
        distances = map(lambda x: objective(x[0], x[1]), zip(guesses, dataset_expected))
        average = sum(distances) / len(dataset_inputs)
        results.append((network, average))
    # pick top n
    results = sorted(results, key=lambda x: x[1])
    performance_print(results, g)
    results = results[:top_n]
    results = map(lambda x: x[0], results)
    # duplicate
    results = list(itertools.islice(itertools.cycle(results), population))
    # vary
    t = t_at(g)
    for i in range(top_n, population):
        results[i] = vary_network(results[i], t)
    pop = results
