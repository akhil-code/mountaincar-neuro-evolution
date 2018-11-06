import pickle

from numpy import argmax

import gym
from genetic import Population


class Game:
    def __init__(self):
        # layers used for neural network
        self.layers = (2, 8, 8, 8, 3)
        # initializing population for genetic algorithm
        self.population = Population(layers=self.layers)
        # initializing openai gym environment for cartpole game
        self.env = gym.make('MountainCar-v0')
        self.episode_threshold = 300

    def loop(self):
        """ loops infinitely for training """
        high_score = -999999                     # used to store high score in training so far
        # learn infinitely
        while True:
            # iterating individuals in population
            for individual in self.population.individuals:
                X = self.env.reset()            # Feature vector
                done = False                    # flag set by gym when game is over
                episode_length = 0

                while not done and episode_length < self.episode_threshold:
                    # self.env.render()
                    y = individual.nn.feed_forward(X)
                    X, reward, done, info = self.env.step(argmax(y))
                    individual.score = X[0]
                    episode_length += 1
                
                # save model when new high score is achieved
                if individual.score > high_score:
                    data = {
                        'score' : individual.score,
                        'number_of_layers' : individual.nn.number_of_layers,
                        'weights' : individual.nn.weights,
                    }
                    # write to file
                    with open('model.pkl', 'wb') as f:
                        pickle.dump(data, f)
                    high_score = individual.score
            
            self.population.evolve()

if __name__ == '__main__':
    game = Game()
    game.loop()
