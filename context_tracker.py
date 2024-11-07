import ray


@ray.remote
class ContextTracker:
    def __init__(self):
        self.context = []

    def add_tokens(self, tokens):
        self.context.extend(tokens)

    def get_tokens(self):
        return self.context
