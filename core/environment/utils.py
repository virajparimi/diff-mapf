import numpy as np
import pybullet as p

from core.environment.arm import UR5


class Target:
    NORMAL = 0
    TOUCHED = 1

    def __init__(
        self,
        pose=[(0, 0, 0), (0, 0, 0, 0)],
        radius=0.02,
        mass=0.0,
        color=[0.2, 0.2, 0.2],
    ):
        self.color = color
        self.radius = radius
        self._mode = Target.NORMAL

        self.viz_id = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=0.1,
            rgbaColor=(self.color[0], self.color[1], self.color[2], 0.7),
        )
        self.body_id = p.createMultiBody(
            baseMass=mass,
            basePosition=pose[0],
            baseOrientation=pose[1],
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=self.viz_id,
        )

        self.normal()

    def normal(self):
        if self._mode != Target.NORMAL:
            p.changeVisualShape(
                self.body_id,
                -1,
                rgbaColor=(self.color[0], self.color[1], self.color[2], 0.5),
            )
            self._mode = Target.NORMAL

    def set_pose(self, pose):
        p.resetBasePositionAndOrientation(self.body_id, pose[0], pose[1])

    def touched(self):
        if self._mode != Target.TOUCHED:
            p.changeVisualShape(
                self.body_id,
                -1,
                rgbaColor=(0.4, 1.0, 0.4, 0.8),
            )
            self._mode = Target.TOUCHED


def create_circular_poses(radius, count):
    return [
        [
            [
                radius * np.cos(2 * np.pi * i / count),
                radius * np.sin(2 * np.pi * i / count),
                0,
            ],
            p.getQuaternionFromEuler([0, 0, 2 * np.pi * i / count]),
        ]
        for i in range(count)
    ]


def create_arms(radius, count, speed, training=True, arm_type="UR5"):
    if arm_type == "UR5":
        return [
            UR5(pose=pose, enabled=False, velocity=speed, training=training)
            for pose in create_circular_poses(radius, count)
        ]
    else:
        raise ValueError(f"Invalid arm type: {arm_type}")


def position_to_high_frequency_position(position):
    position_high_frequency = []
    for i in range(1, 11):
        frequency = 2 * np.pi**i
        position_high_frequency.extend(
            [
                np.sin(frequency * position[0]),
                np.sin(frequency * position[1]),
                np.sin(frequency * position[2]),
            ]
        )
        position_high_frequency.extend(
            [
                np.cos(frequency * position[0]),
                np.cos(frequency * position[1]),
                np.cos(frequency * position[2]),
            ]
        )
    return position_high_frequency


def pose_to_high_frequency_pose(pose):
    position, orientation = pose
    return position_to_high_frequency_position(position), orientation
