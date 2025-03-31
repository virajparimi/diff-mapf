import numpy as np
import pybullet as p
import pybullet_data
from itertools import product
from collections import namedtuple

BASE_LINK = -1
CIRCULAR_LIMITS = (-np.pi, np.pi)

JointInfo = namedtuple(
    "JointInfo",
    [
        "jointIndex",
        "jointName",
        "jointType",
        "qIndex",
        "uIndex",
        "flags",
        "jointDamping",
        "jointFriction",
        "jointLowerLimit",
        "jointUpperLimit",
        "jointMaxForce",
        "jointMaxVelocity",
        "linkName",
        "jointAxis",
        "parentFramePos",
        "parentFrameOrn",
        "parentIndex",
    ],
)

JointState = namedtuple(
    "JointState",
    [
        "jointPosition",
        "jointVelocity",
        "jointReactionForces",
        "appliedJointMotorTorque",
    ],
)

LinkState = namedtuple(
    "LinkState",
    [
        "linkWorldPosition",
        "linkWorldOrientation",
        "localInertialFramePosition",
        "localInertialFrameOrientation",
        "worldLinkFramePosition",
        "worldLinkFrameOrientation",
    ],
)


def get_joint_info(body, joint):
    return JointInfo(*p.getJointInfo(body, joint))


def get_joint_limits(body, joint):
    if is_circular(body, joint):
        return CIRCULAR_LIMITS
    joint_info = get_joint_info(body, joint)
    return joint_info.jointLowerLimit, joint_info.jointUpperLimit


def get_num_joints(body):
    return p.getNumJoints(body)


def get_joints(body):
    return list(range(get_num_joints(body)))


def get_joint_state(body, joint):
    return JointState(*p.getJointState(body, joint))


def get_joint_position(body, joint):
    return get_joint_state(body, joint).jointPosition


def get_joint_positions(body, joints=None):
    assert joints is not None
    return list(get_joint_position(body, joint) for joint in joints)


def set_joint_position(body, joint, value):
    p.resetJointState(body, joint, value)


def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        set_joint_position(body, joint, value)


def control_joints(body, joints, positions, velocity=1.0, acceleration=2.0):
    return p.setJointMotorControlArray(
        body,
        joints,
        p.POSITION_CONTROL,
        targetPositions=positions,
        positionGains=[velocity] * len(joints),
    )


get_links = get_joints


def is_circular(body, joint):
    joint_info = get_joint_info(body, joint)
    if joint_info.jointType == p.JOINT_FIXED:
        return False
    return joint_info.jointUpperLimit < joint_info.jointLowerLimit


def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def circular_difference(theta2, theta1):
    return wrap_angle(theta2 - theta1)


def get_all_link_parents(body):
    parents = {}
    for link in get_links(body):
        parents[link] = get_link_parent(body, link)
    return parents


def get_all_link_children(body):
    children = {}
    for child, parent in get_all_link_parents(body).items():
        if parent not in children:
            children[parent] = []
        children[parent].append(child)
    return children


def get_link_children(body, link):
    children = get_all_link_children(body)
    return children.get(link, [])


def get_link_descendants(body, link):
    descendants = []
    for child in get_link_children(body, link):
        descendants.append(child)
        descendants += get_link_descendants(body, child)
    return descendants


def get_moving_links(body, moving_joints):
    moving_links = list(moving_joints)
    for link in moving_joints:
        moving_links += get_link_descendants(body, link)
    return list(set(moving_links))


def get_link_parent(body, link):
    if link == BASE_LINK:
        return None
    return get_joint_info(body, link).parentIndex


def get_link_ancestors(body, link):
    parent = get_link_parent(body, link)
    if parent is None:
        return []
    return get_link_ancestors(body, parent) + [parent]


def get_body_pose(body):
    raw = p.getBasePositionAndOrientation(body)
    position = list(raw[0])
    orientation = list(raw[1])
    return [position, orientation]


def get_link_state(body, link):
    return LinkState(*p.getLinkState(body, link))


def get_link_pose(body, link):
    if link == BASE_LINK:
        return get_body_pose(body)
    link_state = get_link_state(body, link)
    return [
        list(link_state.worldLinkFramePosition),
        list(link_state.worldLinkFrameOrientation),
    ]


def get_joint_ancestors(body, link):
    return get_link_ancestors(body, link) + [link]


def get_moving_pairs(body, moving_joints):
    moving_links = get_moving_links(body, moving_joints)
    for i in range(len(moving_links)):
        link1 = moving_links[i]
        ancestors1 = set(get_joint_ancestors(body, link1)) & set(moving_joints)
        for j in range(i + 1, len(moving_links)):
            link2 = moving_links[j]
            ancestors2 = set(get_joint_ancestors(body, link2)) & set(moving_joints)
            if ancestors1 != ancestors2:
                yield link1, link2


def are_links_adjacent(body, link1, link2):
    return (get_link_parent(body, link1) == link2) or (
        get_link_parent(body, link2) == link1
    )


def get_self_link_pairs(body, joints, disabled_collisions=set()):
    moving_links = get_moving_links(body, joints)
    fixed_links = list(set(get_links(body)) - set(moving_links))
    check_link_pairs = list(product(moving_links, fixed_links)) + list(
        get_moving_pairs(body, joints)
    )
    check_link_pairs = list(
        filter(lambda pair: not are_links_adjacent(body, *pair), check_link_pairs)
    )
    check_link_pairs = list(
        filter(
            lambda pair: (pair not in disabled_collisions)
            and (pair[::-1] not in disabled_collisions),
            check_link_pairs,
        )
    )
    return check_link_pairs


def get_difference_fn(body, joints):
    def fn(q2, q1):
        difference = []
        for joint, value2, value1 in zip(joints, q2, q1):
            difference.append(
                circular_difference(value2, value1)
                if is_circular(body, joint)
                else (np.array(value2) - np.array(value1))
            )
        return list(difference)

    return fn


def get_distance_fn(body, joints, weights=None):
    if weights is None:
        weights = np.ones(len(joints))
    difference_fn = get_difference_fn(body, joints)

    def fn(q1, q2):
        diff = np.array(difference_fn(q2, q1))
        return np.sqrt(np.dot(weights, diff * diff))

    return fn


def get_sample_fn(body, joints):
    def fn():
        values = []
        for joint in joints:
            limits = (
                CIRCULAR_LIMITS
                if is_circular(body, joint)
                else get_joint_limits(body, joint)
            )
            values.append(np.random.uniform(*limits))
        return list(values)

    return fn


def get_extend_fn(body, joints, resolutions=None):
    if resolutions is None:
        resolutions = np.ones(len(joints)) * 0.05
    difference_fn = get_difference_fn(body, joints)

    def fn(q1, q2):
        diffs = difference_fn(q2, q1)
        steps = np.abs(np.divide(diffs, resolutions))  # type: ignore
        num_steps = int(max(steps))
        waypoints = []
        for i in range(num_steps):
            waypoints.append(
                list(((float(i) + 1.0) / float(num_steps)) * np.array(diffs) + q1)
            )
        return waypoints

    return fn


def violates_limit(body, joint, value):
    if not is_circular(body, joint):
        lower, upper = get_joint_limits(body, joint)
        if value < lower or value > upper:
            return True
    return False


def violates_limits(body, joints, values):
    return any(
        violates_limit(body, joint, value) for joint, value in zip(joints, values)
    )


def reset_camera(yaw=50.0, pitch=-35.0, dist=5.0, target=(0.0, 0.0, 0.0)):
    p.resetDebugVisualizerCamera(
        cameraDistance=dist,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=target,
    )


def configure_pybullet(
    debug=False,
    yaw=50.0,
    pitch=-35.0,
    dist=1.2,
    target=(0.0, 0.0, 0.0),
):
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    if not debug:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
    reset_camera(yaw=yaw, pitch=pitch, dist=dist, target=target)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
    p.resetSimulation()
    p.setRealTimeSimulation(0)
    p.setGravity(0, 0, -9.81)


def remove_all_markers():
    p.removeAllUserDebugItems()


def draw_line(start, end, rgb_color=(1, 0, 0), width=3, lifetime=0):
    line_id = p.addUserDebugLine(
        lineFromXYZ=start,
        lineToXYZ=end,
        lineColorRGB=rgb_color,
        lineWidth=width,
        lifeTime=lifetime,
    )
    return line_id


def forward_kinematics(body, joints, positions, eef_link=None):
    eef_link = get_num_joints(body) - 1 if eef_link is None else eef_link
    old_positions = get_joint_positions(body, joints)
    set_joint_positions(body, joints, positions)
    eef_pose = get_link_pose(body, eef_link)
    set_joint_positions(body, joints, old_positions)
    return eef_pose


def inverse_kinematics(body, eef_link, position, orientation=None):
    if orientation is None:
        joint_values = p.calculateInverseKinematics(
            bodyUniqueId=body,
            endEffectorLinkIndex=eef_link,
            targetPosition=position,
            residualThreshold=1e-3,
        )
    else:
        joint_values = p.calculateInverseKinematics(
            bodyUniqueId=body,
            endEffectorLinkIndex=eef_link,
            targetPosition=position,
            targetOrientation=orientation,
            residualThreshold=1e-3,
        )
    return joint_values
