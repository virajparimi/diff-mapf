import os
import pickle
import numpy as np
import pybullet as p
from urdfpy import URDF
from transforms3d.affines import decompose
from transforms3d.quaternions import mat2quat
from os.path import abspath, dirname, basename, splitext


class PybulletRecorder:
    class LinkTracker:
        def __init__(
            self, name, body_id, link_id, link_origin, mesh_path, mesh_scale, color
        ):
            self.name = name
            self.color = color
            self.body_id = body_id
            self.link_id = link_id
            self.mesh_path = mesh_path
            self.mesh_scale = mesh_scale
            decomposed_origin = decompose(link_origin)
            orientation = mat2quat(decomposed_origin[1])
            orientation = [
                orientation[1],
                orientation[2],
                orientation[3],
                orientation[0],
            ]
            self.link_pose = [decomposed_origin[0], orientation]

        def transform(self, position, orientation):
            return p.multiplyTransforms(
                position, orientation, self.link_pose[0], self.link_pose[1]
            )

        def get_keyframe(self):
            if self.link_id == -1:
                position, orientation = p.getBasePositionAndOrientation(self.body_id)
                position, orientation = self.transform(
                    position=position, orientation=orientation
                )
            else:
                link_state = p.getLinkState(
                    self.body_id, self.link_id, computeForwardKinematics=True
                )
                position, orientation = self.transform(
                    position=link_state[4], orientation=link_state[5]
                )
            return {
                "position": list(position),
                "orientation": list(orientation),
            }

    def __init__(self):
        self.links = []
        self.states = []

    def register_visual_only(
        self, name, body_id, mesh_path, color=None, mesh_scale=[1, 1, 1]
    ):
        full_path = abspath(mesh_path)
        dir_path = dirname(full_path)
        file_name = splitext(basename(full_path))[0]
        link_origin = np.eye(4)
        self.links.append(
            self.LinkTracker(
                name=name,
                body_id=body_id,
                link_id=-1,
                link_origin=link_origin,
                mesh_path=f"{dir_path}/{file_name}.obj",
                mesh_scale=mesh_scale,
                color=color,
            )
        )

    def register_object(self, body_id, urdf_path, color=None, global_scaling=1):
        body_name = p.getBodyInfo(body_id)[0].decode("gb2312")
        link_id_map = {body_name: -1}
        num_joints = p.getNumJoints(body_id)
        for j in range(num_joints):
            joint_name = p.getJointInfo(body_id, j)[12].decode("gb2312")
            link_id_map[joint_name] = j

        dir_path = dirname(abspath(urdf_path))
        file_name = splitext(basename(urdf_path))[0]
        robot = URDF.load(urdf_path)
        for link in robot.links:
            link_id = link_id_map.get(link.name)
            if link_id is None:
                continue
            if link.visuals:
                for i, link_visual in enumerate(link.visuals):
                    if not (link_visual.geometry and link_visual.geometry.mesh):
                        continue
                    if link_visual.geometry.mesh.scale is None:
                        mesh_scale = [global_scaling] * 3
                    else:
                        mesh_scale = [
                            s * global_scaling for s in link_visual.geometry.mesh.scale
                        ]

                    base_origin = (
                        np.linalg.inv(link.inertial.origin)
                        if link_id == -1
                        else np.identity(4)
                    )
                    combined_origin = base_origin @ link_visual.origin * global_scaling

                    mesh_path = os.path.join(
                        dir_path, link_visual.geometry.mesh.filename
                    )
                    name = f"{file_name}_{body_id}_{link.name}_{i}"

                    self.links.append(
                        self.LinkTracker(
                            name=name,
                            body_id=body_id,
                            link_id=link_id,
                            link_origin=combined_origin,
                            mesh_path=mesh_path,
                            mesh_scale=mesh_scale,
                            color=color,
                        )
                    )

    def add_keyframe(self):
        current_state = {}
        for link in self.links:
            current_state[link.name] = link.get_keyframe()
        self.states.append(current_state)

    def reset(self):
        self.states = []

    def get_formatted_output(self):
        formatted_output = {}
        for link in self.links:
            formatted_output[link.name] = {
                "type": "mesh",
                "mesh_path": link.mesh_path,
                "mesh_scale": link.mesh_scale,
                "frames": [state[link.name] for state in self.states],
                "color": getattr(link, "color", [0.5, 0.5, 0.5]),
            }
        return formatted_output

    def save(self, path):
        if path is None:
            raise ValueError("[Recorder] Save path cannot be None")
        else:
            os.makedirs(dirname(path), exist_ok=True)
            pickle.dump(self.get_formatted_output(), open(path, "wb"))
