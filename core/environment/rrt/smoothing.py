import random


def smooth_path(path, extend_fn, collision_fn, iterations=50):
    smoothed_path = path
    for _ in range(iterations):
        if len(smoothed_path) < 3:
            return smoothed_path

        i = random.randint(0, len(smoothed_path) - 1)
        j = random.randint(0, len(smoothed_path) - 1)

        if abs(i - j) < 2:
            continue

        if i > j:
            i, j = j, i

        shortcut = list(extend_fn(smoothed_path[i], smoothed_path[j]))
        if len(shortcut) < j - i and all(not collision_fn(q) for q in shortcut):
            smoothed_path = (
                smoothed_path[: i + 1] + shortcut + smoothed_path[j + 1 :]  # noqa
            )
    return smoothed_path
