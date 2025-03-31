from time import time

from core.environment.rrt.rrt import TreeNode, configs
from core.environment.rrt.smoothing import smooth_path
from core.environment.rrt.pybullet_utils import draw_line
from core.environment.rrt.rrt_utils import irange, argmin


def direct_path(q1, q2, extend_fn, collision_fn):
    if collision_fn(q1) or collision_fn(q2):
        return None
    path = [q1]
    for q in extend_fn(q1, q2):
        if collision_fn(q):
            return None
        path.append(q)
    return path


def rrt_connect(
    q1,
    q2,
    distance_fn,
    sample_fn,
    extend_fn,
    collision_fn,
    iterations,
    visualize,
    fk,
    group,
    greedy,
    timeout,
):
    start_time = time()
    if collision_fn(q1) or collision_fn(q2):
        return None

    root1, root2 = TreeNode(q1), TreeNode(q2)
    nodes1, nodes2 = [root1], [root2]

    if visualize:
        color1, color2 = [0, 1, 0], [0, 0, 1]

    for _ in irange(iterations):
        if float(time() - start_time) > timeout:
            break

        if len(nodes1) > len(nodes2):
            nodes1, nodes2 = nodes2, nodes1
            if visualize:
                color1, color2 = color2, color1

        sample = sample_fn()

        last1 = argmin(lambda node: distance_fn(node.config, sample), nodes1)
        for q in extend_fn(last1.config, sample):
            if collision_fn(q):
                break
            if visualize:
                assert fk is not None
                if group:
                    for pose_now, pose_prev in zip(fk(q), fk(last1.config)):
                        draw_line(pose_prev[0], pose_now[0], rgb_color=color1, width=1)
                else:
                    pose_now = fk(q)[0]
                    pose_prev = fk(last1.config)[0]
                    draw_line(pose_prev, pose_now, rgb_color=color1, width=1)
            last1 = TreeNode(q, parent=last1)
            nodes1.append(last1)
            if not greedy:
                break

        last2 = argmin(lambda node: distance_fn(node.config, last1.config), nodes2)
        for q in extend_fn(last2.config, last1.config):
            if collision_fn(q):
                break
            if visualize:
                assert fk is not None
                if group:
                    for pose_now, pose_prev in zip(fk(q), fk(last2.config)):
                        draw_line(pose_prev[0], pose_now[0], rgb_color=color2, width=1)
                else:
                    pose_now = fk(q)[0]
                    pose_prev = fk(last2.config)[0]
                    draw_line(pose_prev, pose_now, rgb_color=color2, width=1)
            last2 = TreeNode(q, parent=last2)
            nodes2.append(last2)
            if not greedy:
                break

        if last2.config == last1.config:
            path1, path2 = last1.retrace(), last2.retrace()
            if path1[0] != root1:
                path1, path2 = path2, path1
            return configs(path1[:-1] + path2[::-1])

    return None


def birrt(
    start_config,
    goal_config,
    distance_fn,
    sample_fn,
    extend_fn,
    collision_fn,
    iterations,
    smooth,
    visualize,
    fk,
    group,
    greedy,
    timeout=100000,
):
    path = direct_path(start_config, goal_config, extend_fn, collision_fn)
    if path is not None:
        return path
    path = rrt_connect(
        q1=start_config,
        q2=goal_config,
        extend_fn=extend_fn,
        sample_fn=sample_fn,
        distance_fn=distance_fn,
        collision_fn=collision_fn,
        fk=fk,
        group=group,
        greedy=greedy,
        timeout=timeout,
        iterations=iterations,
        visualize=visualize,
    )
    if path is not None:
        return smooth_path(path, extend_fn, collision_fn, iterations=smooth)
    return None
