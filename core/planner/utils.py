import quaternion
import numpy as np


def get_pose_residuals(poseA, poseB):
    positionA = np.array(poseA[0])
    xA, yA, zA, wA = poseA[1]
    orientationA = quaternion.quaternion(wA, xA, yA, zA)

    positionB = np.array(poseB[0])
    xB, yB, zB, wB = poseB[1]
    orientationB = quaternion.quaternion(wB, xB, yB, zB)  # type: ignore

    position_residual = np.linalg.norm(positionA - positionB)
    orientation_residual = (orientationA * orientationB.inverse()).angle()
    orientation_residual = orientation_residual % (2 * np.pi)
    if orientation_residual > np.pi:
        orientation_residual = 2 * np.pi - orientation_residual

    return position_residual, orientation_residual
