import os

import numpy

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class StartScreen(object):
    def __init__(self, model):
        self.model = model