# Customized individual class

from deap import base, gp

class FitnessMax(base.Fitness):
    weights = (1.0,)

class Individual(gp.PrimitiveTree):
    def __init__(self, content):
        super().__init__(content)
        self.fitness = FitnessMax()
