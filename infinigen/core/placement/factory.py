# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Alexander Raistrick: AssetFactory, make_asset_collection
# - Lahav Lipson: quickly_resample


import logging
import typing
from collections import OrderedDict

import bpy
import numpy as np
from tqdm import trange

from infinigen.assets.utils.object import center
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed, int_hash

from . import detail

logger = logging.getLogger(__name__)

_LAYOUT_ASSET_CACHE = OrderedDict()
_LAYOUT_ASSET_CACHE_MAX_ENTRIES = 64
_LAYOUT_ASSET_CACHE_HITS = 0
_LAYOUT_ASSET_CACHE_STORES = 0


def _layout_asset_key(gen, inst_seed):
    return (gen.__class__, gen.factory_seed, int(inst_seed))


def cache_layout_asset(gen, inst_seed, obj):
    global _LAYOUT_ASSET_CACHE_STORES
    key = _layout_asset_key(gen, inst_seed)
    old = _LAYOUT_ASSET_CACHE.pop(key, None)
    if old is not None and old != obj:
        butil.delete(list(butil.iter_object_tree(old)))
    _LAYOUT_ASSET_CACHE[key] = obj
    _LAYOUT_ASSET_CACHE_STORES += 1
    while len(_LAYOUT_ASSET_CACHE) > _LAYOUT_ASSET_CACHE_MAX_ENTRIES:
        _, evicted = _LAYOUT_ASSET_CACHE.popitem(last=False)
        butil.delete(list(butil.iter_object_tree(evicted)))
    logger.debug(f"Cached layout asset {key=} entries={len(_LAYOUT_ASSET_CACHE)}")


def pop_cached_layout_asset(gen, inst_seed):
    global _LAYOUT_ASSET_CACHE_HITS
    key = _layout_asset_key(gen, inst_seed)
    obj = _LAYOUT_ASSET_CACHE.pop(key, None)
    if obj is not None:
        _LAYOUT_ASSET_CACHE_HITS += 1
        logger.debug(f"Reused layout asset {key=} entries={len(_LAYOUT_ASSET_CACHE)}")
    return obj


def clear_layout_asset_cache():
    global _LAYOUT_ASSET_CACHE_HITS, _LAYOUT_ASSET_CACHE_STORES
    entries = len(_LAYOUT_ASSET_CACHE)
    for obj in list(_LAYOUT_ASSET_CACHE.values()):
        butil.delete(list(butil.iter_object_tree(obj)))
    _LAYOUT_ASSET_CACHE.clear()
    logger.info(
        f"Cleared layout asset cache: entries={entries} "
        f"stores={_LAYOUT_ASSET_CACHE_STORES} hits={_LAYOUT_ASSET_CACHE_HITS}"
    )
    _LAYOUT_ASSET_CACHE_HITS = 0
    _LAYOUT_ASSET_CACHE_STORES = 0


class AssetFactory:
    def __init__(self, factory_seed=None, coarse=False):
        self.factory_seed = factory_seed
        if self.factory_seed is None:
            self.factory_seed = np.random.randint(1e9)

        self.coarse = coarse

        logger.debug(f"{self}.__init__()")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.factory_seed})"

    @staticmethod
    def quickly_resample(obj):
        assert obj.type == "EMPTY", obj.type
        obj.rotation_euler[2] = np.random.uniform(-np.pi, np.pi)

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        # Optionally, override this function to decide what will be used as a placeholder for your asset
        return butil.spawn_cube(size=2)

    def finalize_placeholders(self, placeholders: typing.List[bpy.types.Object]):
        # Optionally, override this function to perform any operations on all the placeholders at once
        # eg joint space colonization, placing vines between trees
        pass

    def asset_parameters(self, distance: float, vis_distance: float) -> dict:
        # Optionally, override to determine the **params input of create_asset w.r.t. camera distance
        return {
            "face_size": detail.target_face_size(distance),
            "distance": distance,
            "vis_distance": vis_distance,
        }

    def create_asset(self, **params) -> bpy.types.Object:
        # Override this function to produce a high detail asset
        raise NotImplementedError

    def finalize_assets(self, assets):
        # Optionally, override this function to perform any operations on all the assets at once
        # eg any cleanup / grouping
        pass

    def spawn_placeholder(self, i, loc, rot):
        # Not intended to be overridden - override create_placeholder instead

        logger.debug(f"{self}.spawn_placeholder({i}...)")

        with FixedSeed(int_hash((self.factory_seed, i))):
            obj = self.create_placeholder(i=i, loc=loc, rot=rot)

        has_sensitive_constraint = any(
            c.type in ["FOLLOW_PATH"] for c in obj.constraints
        )

        if not has_sensitive_constraint:
            obj.location = loc
            obj.rotation_euler = rot
        else:
            logger.debug(
                f"Not assigning placeholder {obj.name=} location due to presence of"
                "location-sensitive constraint, typically a follow curve"
            )
        obj.name = f"{repr(self)}.spawn_placeholder({i})"

        if obj.parent is not None:
            logger.warning(
                f"{obj.name=} has no-none parent {obj.parent.name=}, this may cause it not to get populated"
            )

        return obj

    def spawn_asset(
        self,
        i,
        placeholder=None,
        distance=None,
        vis_distance=0,
        loc=(0, 0, 0),
        rot=(0, 0, 0),
        **kwargs,
    ):
        if not isinstance(i, int):
            raise TypeError(f"{i=} {type(i)=}, expected int")
        # Not intended to be overridden - override create_asset instead

        logger.debug(f"{self}.spawn_asset({i}...)")

        if distance is None:
            distance = detail.scatter_res_distance()

        if self.coarse:
            raise ValueError(
                "Attempted to spawn_asset() on an AssetFactory(coarse=True)"
            )

        if placeholder is None:
            cached = pop_cached_layout_asset(self, i)
            if cached is not None:
                for child in butil.iter_object_tree(cached):
                    child.hide_render = False
                    child.hide_viewport = False
                cached.parent = None
                cached.location = loc
                cached.rotation_euler = rot
                cached.name = f"{repr(self)}.spawn_asset({i})"
                return cached

        user_provided_placeholder = placeholder is not None

        if user_provided_placeholder:
            assert loc == (0, 0, 0) and rot == (0, 0, 0)
        else:
            placeholder = self.spawn_placeholder(i=i, loc=loc, rot=rot)
            self.finalize_placeholders([placeholder])

        gc_targets = [
            bpy.data.meshes,
            bpy.data.textures,
            bpy.data.node_groups,
            bpy.data.materials,
        ]

        export_path = None
        with (
            FixedSeed(int_hash((self.factory_seed, i))),
            butil.GarbageCollect(gc_targets, verbose=False),
        ):
            params = self.asset_parameters(distance, vis_distance)
            params.update(kwargs)
            obj = self.create_asset(i=i, placeholder=placeholder, **params)
            # TODO: clean this up
            if "export" in params and params["export"]:
                obj, export_path, semantic_mapping = obj
                assert export_path

        obj.name = f"{repr(self)}.spawn_asset({i})"

        if user_provided_placeholder:
            if obj is not placeholder:
                if obj.parent is None:
                    butil.parent_to(obj, placeholder, no_inverse=True)
            else:
                obj.hide_render = False
        else:
            obj.parent = None
            obj.location = placeholder.location
            obj.rotation_euler = placeholder.rotation_euler
            butil.delete(placeholder)

        if export_path is not None:
            return obj, export_path, semantic_mapping
        return obj

    __call__ = spawn_asset  # for convinience

    def post_init(self):
        pass


def make_asset_collection(
    spawn_fns,
    n,
    name=None,
    weights=None,
    as_list=False,
    verbose=True,
    centered=False,
    **kwargs,
):
    if not isinstance(spawn_fns, list):
        spawn_fns = [spawn_fns]
    if weights is None:
        weights = np.ones(len(spawn_fns))
    weights /= sum(weights)

    if name is None:
        name = ",".join([repr(f) for f in spawn_fns])

    if verbose:
        logger.info(f"Generating collection of {n} assets from {name}")

    objs = [[] for _ in range(len(spawn_fns))]
    r = trange(n) if verbose else range(n)
    for i in r:
        fn_idx = np.random.choice(np.arange(len(spawn_fns)), p=weights)
        obj = spawn_fns[fn_idx](i=i, **kwargs)
        if centered:
            obj.location = -center(obj)
            butil.apply_transform(obj, True)
        objs[fn_idx].append(obj)

    for os, f in zip(objs, spawn_fns):
        if hasattr(f, "finalize_assets"):
            f.finalize_assets(os)

    objs = sum(objs, start=[])

    if as_list:
        return objs
    else:
        col = butil.group_in_collection(objs, name=f"assets:{name}", reuse=False)
        col.hide_viewport = True
        col.hide_render = True
        return col
