import numpy as np
import ray


@ray.remote
class ContextTracker:
    def __init__(self):
        self.context = []
        self.audio = np.array([])

    def add_tokens(self, tokens):
        self.context.extend(tokens)

    def get_tokens(self):
        return self.context

    def last_sample(self, audio):
        self.audio = audio

    def get_last(self):
        return self.audio