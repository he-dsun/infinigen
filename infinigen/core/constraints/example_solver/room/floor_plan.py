# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei

import time

import gin
import numpy as np
import shapely
from numpy.random import uniform
from shapely.affinity import translate
from tqdm import tqdm, trange

from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints.evaluator.evaluate import evaluate_problem
from infinigen.core.constraints.example_solver.state_def import (
    ObjectState,
    RelationState,
    State,
)
from infinigen.core.tags import Semantics
from infinigen.core.util.math import FixedSeed

from .base import RoomGraph, room_level, room_name, room_type
from .contour import ContourFactory
from .graph import GraphMaker
from .segment import SegmentMaker
from .solidifier import BlueprintSolidifier
from .solver import FloorPlanMoves


@gin.configurable
class FloorPlanSolver:
    def __init__(
        self,
        factory_seed,
        consgraph,
        n_divide_trials=100,
        iters_mult=200,
        max_graph_trials=100,
        max_contour_attempts=100,
        max_solve_attempts=20,
        max_seconds=180.0,
        single_room_fast_path=False,
        single_room_type=None,
    ):
        self.factory_seed = factory_seed
        self.max_contour_attempts = max(1, int(max_contour_attempts))
        self.max_solve_attempts = max(1, int(max_solve_attempts))
        self.max_seconds = None
        if max_seconds is not None and float(max_seconds) > 0:
            self.max_seconds = float(max_seconds)
        self.single_room_fast_path = bool(single_room_fast_path)
        self.single_room_type = single_room_type
        self.single_room_state = None
        with FixedSeed(factory_seed):
            self.constants = consgraph.constants
            self.consgraph = consgraph
            self.n_stories = self.constants.n_stories
            self.fixed_contour = self.constants.fixed_contour
            self.contour_factory = ContourFactory(self.consgraph)
            self.n_trials = max(1, int(max_graph_trials))
            self.graphs = []
            self.widths, self.heights = [], []
            self.contours = []

            if self.single_room_fast_path:
                self._build_single_room()
                self.segment_makers = []
            else:
                self.build_graphs()
                self.segment_makers = [
                    SegmentMaker(
                        factory_seed,
                        self.constants,
                        consgraph,
                        self.contours[i],
                        self.graphs[i],
                        i,
                    )
                    for i in range(self.n_stories)
                ]

            self.solver = FloorPlanMoves(self.constants)
            self.solidifiers = [
                BlueprintSolidifier(consgraph, g, i) for i, g in enumerate(self.graphs)
            ]

            self.n_divide_trials = n_divide_trials
            self.iter_per_room = iters_mult
            self.score_scale = 5
            self.staircase_solver_prob = 0.1

    def _build_single_room(self):
        if self.single_room_type is None:
            room_semantic = np.random.choice(
                [
                    Semantics.Bathroom,
                    Semantics.Bedroom,
                    Semantics.DiningRoom,
                    Semantics.Kitchen,
                    Semantics.LivingRoom,
                ]
            )
        elif isinstance(self.single_room_type, Semantics):
            room_semantic = self.single_room_type
        else:
            room_type_name = str(self.single_room_type)
            try:
                room_semantic = Semantics[room_type_name]
            except KeyError:
                room_semantic = Semantics(room_type_name.lower())

        self.n_stories = 1
        room_key = room_name(room_semantic, 0)
        exterior_key = room_name(Semantics.Exterior, 0)
        graph = RoomGraph([[1], [0]], [room_key, exterior_key], entrance=0)
        graph_maker = GraphMaker(
            self.factory_seed,
            self.consgraph,
            0,
            fast=True,
        )
        width, height = graph_maker.suggest_dimensions(graph)
        contour = shapely.box(0, 0, width, height)
        room_polygon = self.constants.canonicalize(contour)
        shared_edge = shapely.MultiLineString([contour.boundary])

        state = State(graphs=[graph])
        state.objs[room_key] = ObjectState(
            polygon=room_polygon,
            relations=[],
            tags={
                room_semantic,
                Semantics.RoomContour,
                Semantics.GroundFloor,
            },
        )
        state.objs[exterior_key] = ObjectState(
            polygon=contour,
            relations=[
                RelationState(
                    cl.SharedEdge(),
                    room_key,
                    value=shared_edge,
                )
            ],
            tags={Semantics.Exterior, Semantics.RoomContour},
        )

        self.graphs = [graph]
        self.contours = [contour]
        self.widths = [width]
        self.heights = [height]
        self.single_room_state = state

    def _check_timeout(self, started, stage):
        if (
            self.max_seconds is not None
            and time.monotonic() - started > self.max_seconds
        ):
            raise TimeoutError(
                f"FloorPlanSolver.{stage} exceeded {self.max_seconds:.1f}s"
            )

    def build_graphs(self):
        started = time.monotonic()
        for i in range(self.n_stories):
            self._check_timeout(started, "build_graphs")
            graph_maker = GraphMaker(self.factory_seed, self.consgraph, i)
            if self.fixed_contour and i > 0:
                width, height = self.widths[-1], self.heights[-1]
                graph = graph_maker.make_graph(np.random.randint(1e6))
            else:
                for j in range(self.n_trials):
                    graph = graph_maker.make_graph(np.random.randint(1e6))
                    args = (
                        [self.widths[-1], self.heights[-1]]
                        if len(self.graphs) > 0
                        else [None, None]
                    )
                    width, height = graph_maker.suggest_dimensions(graph, *args)
                    if width is not None and height is not None:
                        break
                else:
                    raise Exception("Invalid graph")
            self.graphs.append(graph)
            for _attempt in range(self.max_contour_attempts):
                self._check_timeout(started, "build_graphs")
                for j in range(self.n_trials):
                    if self.fixed_contour and i > 0:
                        self.contours.append(self.contours[-1])
                        break
                    contour = shapely.box(0, 0, width, height)
                    if len(self.contours) > 0:
                        x_offset = self.constants.unit_cast(
                            (width - self.widths[0]) * uniform(0, 1)
                        )
                        y_offset = self.constants.unit_cast(
                            (height - self.heights[0]) * uniform(0, 1)
                        )
                        contour = translate(contour, -x_offset, -y_offset)
                        if not self.contours[-1].contains(contour):
                            continue
                    self.contours.append(contour)
                    break
                else:
                    if (
                        width <= self.constants.unit
                        or height <= self.constants.unit
                    ):
                        break
                    if width / height > self.widths[-1] / self.heights[-1]:
                        width -= self.constants.unit
                    else:
                        height -= self.constants.unit
                if len(self.contours) > i:
                    break
            else:
                raise RuntimeError(
                    f"FloorPlanSolver failed to build contour {i} after "
                    f"{self.max_contour_attempts} attempts"
                )
            if len(self.contours) <= i:
                raise RuntimeError(
                    f"FloorPlanSolver failed to build contour {i}; "
                    "the room dimensions became too small"
                )

            self.widths.append(width)
            self.heights.append(height)

    def solve(self):
        started = time.monotonic()
        if self.single_room_state is None:
            state = State(graphs=self.graphs)
            for _attempt in range(self.max_solve_attempts):
                self._check_timeout(started, "solve")
                pholder = self.contour_factory.add_staircase(self.contours[-1])
                state.objs = {}
                states = []
                for j in range(self.n_stories):
                    for _ in trange(
                        self.n_divide_trials * (j + 1) ** 2,
                        desc=f"Dividing segments for {j}",
                    ):
                        st = self.segment_makers[j].build_segments(pholder)
                        if st is not None:
                            states.append(st)
                            state.objs.update(st.objs)
                            break
                    else:
                        break
                if len(states) == self.n_stories:
                    break
            else:
                raise RuntimeError(
                    f"FloorPlanSolver failed to segment {self.n_stories} stories after "
                    f"{self.max_solve_attempts} attempts"
                )
            state = self.simulated_anneal(state, started=started)
        else:
            state = self.single_room_state
        self.contour_factory.decorate(state)

        obj_states = {}
        for j in range(self.n_stories):
            self._check_timeout(started, "solidify")
            with FixedSeed(self.factory_seed):
                st, rooms_meshed = self.solidifiers[j].solidify(
                    State({k: v for k, v in state.objs.items() if room_level(k) == j})
                )
            obj_states.update(st.objs)
        unique_roomtypes = set()
        for graph in self.graphs:
            for s in graph.names:
                unique_roomtypes.add(Semantics(room_type(s)))
        dimensions = (
            self.widths[0],
            self.heights[0],
            self.constants.wall_height * self.n_stories,
        )
        return State(obj_states), unique_roomtypes, dimensions

    def simulated_anneal(self, state, started=None):
        if started is None:
            started = time.monotonic()
        consgraph = self.consgraph.filter("room")
        consgraph.constraints["graph"] = cl.graph_coherent(self.consgraph.constants)
        score, _ = evaluate_problem(consgraph, state, memo={})
        it = self.iter_per_room * sum(len(g) for g in self.graphs)
        with tqdm(total=it, desc="Sampling solutions") as pbar:
            while pbar.n < it:
                if pbar.n % 50 == 0:
                    self._check_timeout(started, "simulated_anneal")
                state_ = self.solver.perturb_state(state)
                score_, violated_ = evaluate_problem(consgraph, state_, memo={})
                scale = self.score_scale * pbar.n / it
                if np.log(uniform()) < (score - score_) * scale and not violated_:
                    state = state_
                    score = score_
                pbar.update(1)
                pbar.set_postfix(score=score)
        return state
