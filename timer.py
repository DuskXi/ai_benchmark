import time


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = 0

    def start(self):
        self.start_time = time.time()
        return self

    def stop(self):
        self.end_time = time.time()
        self.duration += self.end_time - self.start_time
        return self

    def reset(self):
        self.duration = 0
        return self
