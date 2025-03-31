import time
import heapq
import graphviz
import numpy as np

from core.planner.base_search import BaseSearch


class CBSNode:
    def __init__(self, plan_indices, constraints, cost, node_id, parent_id):
        self.cost = cost
        self.constraints = constraints
        self.plan_indices = plan_indices
        # Useful for debugging
        self.id = node_id
        self.parent_id = parent_id

    def __lt__(self, other):
        return self.cost < other.cost


class ConflictBasedSearch(BaseSearch):
    def __init__(
        self, plans, parameters, planners, sim_steps=1.0, pos_tol=0.02, ori_tol=0.1
    ):
        super().__init__(plans, parameters, planners, sim_steps, pos_tol, ori_tol)
        self.open_set = []

        self.metrics = {
            "num_expanded": 0,
            "num_generated": 0,
            "planning_time": 0.0,
            "num_collision_checks": 0,
        }

        self.pos_tol = pos_tol
        self.ori_tol = ori_tol
        self.sim_steps = sim_steps

        self.node_id_counter = 0
        self.graph = graphviz.Digraph("CBS-Tree", comment="Conflict-Based Search Tree")
        self.graph.attr(rankdir="TB", size="20,20")

    def _add_graph_node(self, node, **kwargs):
        label = (
            f"ID: {node.id}\n"
            f"Cost: {node.cost:.2f}\n"
            f"Plans: {tuple(node.plan_indices)}"
        )
        self.graph.node(name=str(node.id), label=label, **kwargs)

    def _add_graph_edge(self, parent_node, child_node, constraint_info):
        self.graph.edge(str(parent_node.id), str(child_node.id), label=constraint_info)

    def find_plans(self, agents_deque):
        start_time = time.time()
        earliest_collision_time = self.parameters["prediction_horizon"]
        root_plan_indices = [0] * self.num_agents
        root_plan = [self.plans[i][0] for i in range(self.num_agents)]
        root_cost = self.compute_cost(root_plan, root_plan_indices)

        root_id = self.node_id_counter
        self.node_id_counter += 1
        initial_node = CBSNode(
            root_plan_indices, {}, root_cost, node_id=root_id, parent_id=None
        )
        self._add_graph_node(initial_node, style="filled", fillcolor="lightblue", shape="box")
        base_plan = root_plan.copy()

        self.metrics["num_generated"] += 1
        heapq.heappush(self.open_set, initial_node)

        while self.open_set and time.time() - start_time < self.parameters["timeout"]:
            self.metrics["num_expanded"] += 1
            current_node = heapq.heappop(self.open_set)

            if self.graph.body[-(len(str(current_node.id)) + 13) :] != "palegreen":  # NOQA
                self._add_graph_node(
                    current_node, style="filled", fillcolor="lightgrey", shape="box"
                )

            plan = [
                self.plans[i][current_node.plan_indices[i]]
                for i in range(self.num_agents)
            ]

            conflict = None
            conflict_found = False
            for agent_A in range(self.num_agents):
                for agent_B in range(agent_A + 1, self.num_agents):
                    key = tuple(current_node.plan_indices)
                    if key not in self.collision_cache:
                        self.collision_cache[key] = self.check_collisions(
                            (agent_A, plan[agent_A]), (agent_B, plan[agent_B])
                        )
                    if self.collision_cache[key][0]:
                        conflict_found = True
                        conflict = self.collision_cache[key][2]
                        if self.collision_cache[key][1] != -1:
                            if self.collision_cache[key][1] < earliest_collision_time:
                                earliest_collision_time = self.collision_cache[key][1]
                                base_plan = plan.copy()
                        break
                if conflict_found:
                    break

            if not conflict_found:
                if self.num_agents == 1:
                    dummy_plan = np.zeros_like(plan[0])
                    collision, first_collision_step, _ = self.check_collisions(
                        (0, plan[0]), (0, dummy_plan)
                    )
                    earliest_collision_time = (
                        first_collision_step
                        if collision
                        else self.parameters["prediction_horizon"]
                    )

                self._add_graph_node(
                    current_node,
                    style="filled",
                    fillcolor="palegreen",
                    shape="doublecircle",
                )
                if self.node_id_counter > 1:
                    self.graph.render(
                        "cbs_search_tree", view=False, cleanup=True, format="svg"
                    )
                self.metrics["planning_time"] = time.time() - start_time
                return plan, earliest_collision_time

            assert conflict is not None
            agent_A = None
            agent_B = None
            for i in range(self.num_agents):
                if (
                    conflict[0]
                    and self.single_agent_planners[i].pybullet_id == conflict[0]
                ):
                    agent_A = i
                if (
                    conflict[1]
                    and self.single_agent_planners[i].pybullet_id == conflict[1]
                ):
                    agent_B = i
            if agent_A is None or agent_B is None:
                # We are colliding with some object who we cannot control
                return base_plan, earliest_collision_time
            conflict = (agent_A, agent_B)

            updated_constraints = current_node.constraints.copy()
            updated_constraints.setdefault(agent_A, set()).add(
                current_node.plan_indices[agent_A]
            )
            updated_constraints.setdefault(agent_B, set()).add(
                current_node.plan_indices[agent_B]
            )

            for A in range(self.parameters["num_samples"]):
                if A not in updated_constraints[agent_A]:
                    new_indices = current_node.plan_indices.copy()
                    new_indices[agent_A] = A
                    new_plan = [
                        self.plans[k][new_indices[k]] for k in range(self.num_agents)
                    ]
                    new_cost = self.compute_cost(new_plan, new_indices)

                    new_id = self.node_id_counter
                    self.node_id_counter += 1

                    new_node = CBSNode(
                        new_indices,
                        updated_constraints,
                        new_cost,
                        node_id=new_id,
                        parent_id=current_node.id,
                    )

                    constraint_label = (
                        f"Agent {agent_A} != Plan {current_node.plan_indices[agent_A]}"
                    )
                    self._add_graph_node(new_node, shape="box")
                    self._add_graph_edge(current_node, new_node, constraint_label)

                    self.metrics["num_generated"] += 1
                    heapq.heappush(self.open_set, new_node)

            for B in range(self.parameters["num_samples"]):
                if B not in updated_constraints[agent_B]:
                    new_indices = current_node.plan_indices.copy()
                    new_indices[agent_B] = B
                    new_plan = [
                        self.plans[k][new_indices[k]] for k in range(self.num_agents)
                    ]
                    new_cost = self.compute_cost(new_plan, new_indices)

                    new_id = self.node_id_counter
                    self.node_id_counter += 1
                    new_node = CBSNode(
                        new_indices,
                        updated_constraints,
                        new_cost,
                        node_id=new_id,
                        parent_id=current_node.id,
                    )

                    constraint_label = (
                        f"Agent {agent_B} != Plan {current_node.plan_indices[agent_B]}"
                    )
                    self._add_graph_node(new_node, shape="box")
                    self._add_graph_edge(current_node, new_node, constraint_label)

                    self.metrics["num_generated"] += 1
                    heapq.heappush(self.open_set, new_node)

            dual_plan_A = self.dual_agent_planner.predict_plan(conflict, agents_deque)
            if dual_plan_A is not None:
                for A in range(self.parameters["num_samples"]):
                    new_plan_A = dual_plan_A[A]
                    collision, _, _ = self.check_collisions(
                        (agent_A, new_plan_A), (agent_B, plan[agent_B])
                    )
                    if not collision:
                        self.plans[agent_A] = np.concatenate(
                            [self.plans[agent_A], new_plan_A[None, ...]], axis=0
                        )
                        new_indices = current_node.plan_indices.copy()
                        new_indices[agent_A] = len(self.plans[agent_A]) - 1
                        new_plan = [
                            self.plans[k][new_indices[k]]
                            for k in range(self.num_agents)
                        ]
                        new_plan[agent_A] = new_plan_A
                        new_cost = self.compute_cost(new_plan, new_indices)

                        new_id = self.node_id_counter
                        self.node_id_counter += 1

                        new_node = CBSNode(
                            # new_indices, current_node.constraints.copy(), new_cost
                            new_indices,
                            updated_constraints.copy(),
                            new_cost,
                            node_id=new_id,
                            parent_id=current_node.id,
                        )

                        constraint_label = (
                            f"Agent {agent_A} = Dual Plan {new_indices[agent_A]}"
                        )
                        self._add_graph_node(new_node, shape="box")
                        self._add_graph_edge(current_node, new_node, constraint_label)

                        self.metrics["num_generated"] += 1
                        heapq.heappush(self.open_set, new_node)

            conflict = (conflict[1], conflict[0])
            dual_plan_B = self.dual_agent_planner.predict_plan(conflict, agents_deque)
            if dual_plan_B is not None:
                for B in range(self.parameters["num_samples"]):
                    new_plan_B = dual_plan_B[B]
                    collision, _, _ = self.check_collisions(
                        (agent_B, new_plan_B), (agent_A, plan[agent_A])
                    )
                    if not collision:
                        self.plans[agent_B] = np.concatenate(
                            [self.plans[agent_B], new_plan_B[None, ...]], axis=0
                        )
                        new_indices = current_node.plan_indices.copy()
                        new_indices[agent_B] = len(self.plans[agent_B]) - 1
                        new_plan = [
                            self.plans[k][new_indices[k]]
                            for k in range(self.num_agents)
                        ]
                        new_plan[agent_B] = new_plan_B
                        new_cost = self.compute_cost(new_plan, new_indices)

                        new_id = self.node_id_counter
                        self.node_id_counter += 1
                        new_node = CBSNode(
                            # new_indices, current_node.constraints.copy(), new_cost
                            new_indices,
                            updated_constraints.copy(),
                            new_cost,
                            node_id=new_id,
                            parent_id=current_node.id,
                        )

                        constraint_label = (
                            f"Agent {agent_B} = Dual Plan {new_indices[agent_B]}"
                        )
                        self._add_graph_node(new_node, shape="box")
                        self._add_graph_edge(current_node, new_node, constraint_label)

                        self.metrics["num_generated"] += 1
                        heapq.heappush(self.open_set, new_node)

        self.metrics["planning_time"] = time.time() - start_time
        return base_plan, earliest_collision_time
