import numpy as np
import pybullet as p

from core.planner.utils import get_pose_residuals


class BaseSearch:
    def __init__(
        self, plans, parameters, planners, sim_steps=1.0, pos_tol=0.02, ori_tol=0.1
    ):
        self.plans = plans
        self.num_agents = len(plans)
        self.parameters = parameters
        self.collision_cache = {}
        self.dual_agent_planner = planners[1]
        self.single_agent_planners = planners[0]

        self.metrics = {
            "num_expanded": 0,
            "num_generated": 0,
            "planning_time": 0.0,
            "num_collision_checks": 0,
        }

        self.pos_tol = pos_tol
        self.ori_tol = ori_tol
        self.sim_steps = sim_steps

    def compute_cost(self, plans, plan_indices):
        # Plans is vector of length Num-Agents with each entry being (Pred-Horizon, Act-Dim)
        assert self.single_agent_planners is not None, "Planners not set"

        cost = 0.0
        for plan in plans:
            cost += np.sum(np.linalg.norm(plan, axis=1))
        for arm_id in range(self.num_agents):
            ee_pose = self.single_agent_planners[arm_id].arm.get_end_effector_pose()
            target_pose = self.single_agent_planners[
                arm_id
            ].current_task.target_eef_poses[arm_id]
            pos_res, ori_res = get_pose_residuals(target_pose, ee_pose)
            cost += pos_res + ori_res
        for agent_A in range(self.num_agents):
            for agent_B in range(agent_A + 1, self.num_agents):
                key = tuple(plan_indices)
                if key not in self.collision_cache:
                    self.collision_cache[key] = self.check_collisions(
                        (agent_A, plans[agent_A]), (agent_B, plans[agent_B])
                    )
                if self.collision_cache[key][0]:
                    cost += 10.0
        return cost

    def check_collisions(self, arm_A, arm_B):
        self.metrics["num_collision_checks"] += 1
        assert self.single_agent_planners is not None, "Planners not set"
        state_id = p.saveState()

        def adjusted_action(arm, action_idx):
            assert self.single_agent_planners is not None, "Planners not set"

            action = arm[1][action_idx]
            planner = self.single_agent_planners[arm[0]]
            target_pose = planner.current_task.target_eef_poses[arm[0]]
            current_pose = planner.arm.get_end_effector_pose()
            pos_res, ori_res = get_pose_residuals(target_pose, current_pose)
            if pos_res < self.pos_tol and ori_res < self.ori_tol:
                return np.zeros(self.parameters["action_dim"])
            return action

        int_sim_steps = int(self.sim_steps)
        for action_idx in range(self.parameters["prediction_horizon"]):
            action_A = adjusted_action(arm_A, action_idx)
            action_B = adjusted_action(arm_B, action_idx)

            planner_A = self.single_agent_planners[arm_A[0]]
            planner_A.arm.control_arm_joints_delta(action_A)

            planner_B = self.single_agent_planners[arm_B[0]]
            planner_B.arm.control_arm_joints_delta(action_B)

            def get_collision(planner):
                attach_obj = getattr(planner.arm, "attach_object_id", None)
                if attach_obj is not None:
                    return planner.arm.check_collision_with_info(
                        excluded_objects=[attach_obj], self_collision=False
                    )
                return planner.arm.check_collision_with_info()

            for _ in range(int_sim_steps):
                p.stepSimulation()
                collision_A, info_A = get_collision(planner_A)
                collision_B, info_B = get_collision(planner_B)
                if collision_A or collision_B:
                    p.restoreState(state_id)
                    p.removeState(state_id)
                    collision_body_A = info_A[1] if info_A is not None else None
                    collision_body_B = info_B[1] if info_B is not None else None
                    return True, action_idx, (collision_body_A, collision_body_B)

        p.restoreState(state_id)
        p.removeState(state_id)
        return False, -1, (None, None)

    def find_plans(self, agents_deque):
        raise NotImplementedError("find_plans not implemented in BaseSearch")
