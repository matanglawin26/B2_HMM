from itertools import product

'''
References: 
https://towardsdatascience.com/hidden-markov-model-implemented-from-scratch-72865bda430e
https://medium.com/@kangeugine/hidden-markov-model-7681c22f5b9
'''

class Vector:
    def __init__(self, probabilities):
        self.states = list(probabilities.keys())
        self.df = probabilities

class Matrix:
    def __init__(self, probabilities):
        self.states = list(probabilities.keys())
        self.df = probabilities

class HiddenMarkovModel:
    def __init__(self, transition_matrix, emission_matrix, start):
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self.start = start
        self.states = start.states
        self.observables = emission_matrix.states

    def get_probability(self, observations):
        probability = 0
        chain_length = len(observations)
        for chain in product(self.states, repeat=chain_length):
            p_hidden_state = self.start.df[chain[0]]
            p_observations = self.emission_matrix.df[chain[0]].df[observations[0]]

            for i in range(1, chain_length):
                current_state = chain[i]
                previous_state = chain[i-1]
                p_hidden_state *= self.transition_matrix.df[previous_state].df[current_state]
                p_observations *= self.emission_matrix.df[current_state].df[observations[i]]

            probability += p_hidden_state * p_observations

        return probability

    def get_optimal_hidden_sequence(self, observations):
        optimal_sequence = []
        max_probability = 0

        chain_length = len(observations)
        for chain in product(self.states, repeat=chain_length):
            p_hidden_state = self.start.df[chain[0]]
            p_observations = self.emission_matrix.df[chain[0]].df[observations[0]]

            for i in range(1, chain_length):
                current_state = chain[i]
                previous_state = chain[i-1]
                p_hidden_state *= self.transition_matrix.df[previous_state].df[current_state]
                p_observations *= self.emission_matrix.df[current_state].df[observations[i]]

            probability = p_hidden_state * p_observations
            if probability > max_probability:
                max_probability = probability
                optimal_sequence = chain

        return optimal_sequence, max_probability

    def display_probabilities(self, observations_list):
        for observations in observations_list:
            print("Score for {} is {:f}.".format(observations, self.get_probability(observations)))

    def display_optimal_sequence(self, observations_list):
        for observations in observations_list:
            optimal_sequence, max_probability = self.get_optimal_hidden_sequence(observations)

            print("Given the known model and the observation {}, the weather was most likely {} with ~{:0.2f}% probability.".format(observations, optimal_sequence, max_probability * 100))

def main():
    start = Vector({'Rainy': 0.6, 'Sunny': 0.4})

    hidden_state1 = Vector({'Rainy': 0.7, 'Sunny': 0.3})
    hidden_state2 = Vector({'Rainy': 0.4, 'Sunny': 0.6})

    observable_state1 = Vector({'Walk': 0.1, 'Shop': 0.4, 'Clean': 0.5})
    observable_state2 = Vector({'Walk': 0.6, 'Shop': 0.3, 'Clean': 0.1})

    transition_matrix = Matrix({'Rainy': hidden_state1, 'Sunny': hidden_state2})

    emission_matrix = Matrix({'Rainy': observable_state1, 'Sunny': observable_state2})

    hmm = HiddenMarkovModel(transition_matrix, emission_matrix, start)

    observations_list = [
        ['Walk'], ['Clean']
    ]

    hmm.display_probabilities(observations_list)

    observations_list = [
        ['Shop', 'Clean', 'Walk'],
        ['Clean', 'Clean', 'Clean']
    ]

    hmm.display_optimal_sequence(observations_list)


main()
