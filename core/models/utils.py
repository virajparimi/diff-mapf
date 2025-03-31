class EarlyStopping:

    def __init__(self, tolerance, min_delta):
        self.counter = 0
        self.tolerance = tolerance
        self.min_delta = min_delta

    def __call__(self, train_loss, validation_loss):
        if validation_loss - train_loss > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                return True
        else:
            self.counter = 0
        return False
