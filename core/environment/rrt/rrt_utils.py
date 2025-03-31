def irange(start, stop=None, step=1):
    if stop is None:
        stop = start
        start = 0
    while start < stop:
        yield start
        start += step


def argmin(function, sequence):
    values = list(sequence)
    scores = [function(value) for value in values]
    return values[scores.index(min(scores))]
