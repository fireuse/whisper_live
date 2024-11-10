import numpy as np
import ray


@ray.remote
class ContextTracker:
    def __init__(self):
        self.context = []
        self.audio = np.array([])

    def add_tokens(self, tokens):
        self.context.extend(tokens)
        if len(self.context) > 100000:
            self.context = self.context[-1000:]

    def get_tokens(self):
        return self.context

    def last_sample(self, audio):
        self.audio = audio

    def get_last(self):
        return self.audio