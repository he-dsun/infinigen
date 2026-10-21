# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lahav Lipson - Render, flat shading, etc
# - Alex Raistrick - Compositing
# - Hei Law - Initial version

# type: ignore

import json
import logging
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path

import bpy
import gin
import numpy as np
from imageio import imwrite

from infinigen.core import init, surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement import camera as cam_util
from infinigen.core.rendering.post_render import (
    colorize_depth,
    colorize_flow,
    colorize_int_array,
    colorize_normals,
    load_depth,
    load_flow,
    load_normals,
    load_seg_mask,
    load_uniq_inst,
)
from infinigen.core.util.blender import set_geometry_option
from infinigen.core.util.logging import Timer
from infinigen.tools.datarelease_toolkit import reorganize_old_framesfolder
from infinigen.tools.suffixes import get_suffix

TRANSPARENT_SHADERS = {Nodes.TranslucentBSDF, Nodes.TransparentBSDF}
# Material names produced by infinigen/assets/materials that are actually transparent.
TRANSPARENT_MATERIAL_PREFIXES = (
    "shader_glass",
    "shader_translucent_plastic",
    "shader_colored_glass",
    "shader_window_glass",
    "shader_black_glass",
)

# Architectural glass fills the frame but is not an object-level transparent target.
# Keep it out of both the coverage filter and the auto-aim candidates.
ARCHITECTURAL_GLASS_PREFIXES = (
    "WindowFactory",
    "GlassPanelDoorFactory",
    "LiteDoorFactory",
)
GLASS_TARGET_PRIORITY = (
    "AquariumTankFactory",
    "LargePlantContainerFactory",
    "StandingSinkFactory",
)


def _is_architectural_glass(obj):
    return obj.name.startswith(ARCHITECTURAL_GLASS_PREFIXES)


def _glass_priority(obj):
    for i, prefix in enumerate(GLASS_TARGET_PRIORITY):
        if obj.name.startswith(prefix):
            return len(GLASS_TARGET_PRIORITY) - i
    return 0


@contextmanager
def _raycast_visible_assets():
    """Scene ray casts use the viewport depsgraph, where `unique_assets` is hidden during
    rendering (and placeholders may be visible). Toggle them so ray casts see the same
    geometry the renderer does, then restore."""
    toggled = {}
    for name, hidden in (("unique_assets", False), ("placeholders", True)):
        col = bpy.data.collections.get(name)
        if col is not None:
            toggled[name] = col.hide_viewport
            col.hide_viewport = hidden
    bpy.context.view_layer.update()
    try:
        yield
    finally:
        for name, hidden in toggled.items():
            bpy.data.collections[name].hide_viewport = hidden
        bpy.context.view_layer.update()

logger = logging.getLogger(__name__)


def remove_translucency():
    # The asserts were added since these edge cases haven't appeared yet -Lahav
    for material in bpy.data.materials:
        nw = NodeWrangler(material.node_tree)
        for node in nw.nodes:
            if node.bl_idname == Nodes.MixShader:
                fac_soc, shader_1_soc, shader_2_soc = node.inputs
                assert shader_1_soc.is_linked and len(shader_1_soc.links) == 1
                assert shader_2_soc.is_linked and len(shader_2_soc.links) == 1
                shader_1_type = shader_1_soc.links[0].from_node.bl_idname
                shader_2_type = shader_2_soc.links[0].from_node.bl_idname
                assert not (
                    shader_1_type in TRANSPARENT_SHADERS
                    and shader_2_type in TRANSPARENT_SHADERS
                )
                if shader_1_type in TRANSPARENT_SHADERS:
                    assert not fac_soc.is_linked
                    fac_soc.default_value = 1.0
                elif shader_2_type in TRANSPARENT_SHADERS:
                    assert not fac_soc.is_linked
                    fac_soc.default_value = 0.0


def set_pass_indices():
    tree_output = {}
    index = 1
    for obj in bpy.data.objects:
        if obj.hide_render:
            continue
        if obj.pass_index == 0:
            obj.pass_index = index
            index += 1
        object_dict = {"type": obj.type, "object_index": obj.pass_index, "children": []}
        if obj.type == "MESH":
            object_dict["num_verts"] = len(obj.data.vertices)
            object_dict["num_faces"] = len(obj.data.polygons)
            object_dict["materials"] = obj.material_slots.keys()
            object_dict["unapplied_modifiers"] = obj.modifiers.keys()
        tree_output[obj.name] = object_dict
        for child_obj in obj.children:
            if child_obj.pass_index == 0:
                child_obj.pass_index = index
                index += 1
            object_dict["children"].append(child_obj.pass_index)
        index += 1
    return tree_output


def set_material_pass_indices():
    output_material_properties = {}
    mat_index = 1
    for mat in bpy.data.materials:
        if mat.pass_index == 0:
            mat.pass_index = mat_index
            mat_index += 1
        output_material_properties[mat.name] = {"pass_index": mat.pass_index}
    return output_material_properties


# Can be pasted directly into the blender console
def make_clay():
    clay_material = bpy.data.materials.new(name="clay")
    clay_material.diffuse_color = (0.2, 0.05, 0.01, 1)
    for obj in bpy.data.objects:
        if "atmosphere" not in obj.name.lower() and not obj.hide_render:
            if len(obj.material_slots) == 0:
                obj.active_material = clay_material
            else:
                for mat_slot in obj.material_slots:
                    mat_slot.material = clay_material


@gin.configurable
def compositor_postprocessing(
    nw,
    source,
    show=True,
    color_correct=True,
    distort=0,
    glare=False,
):
    if distort > 0:
        source = nw.new_node(
            Nodes.LensDistortion, input_kwargs={"Image": source, "Dispersion": distort}
        )

    if color_correct:
        source = nw.new_node(
            Nodes.BrightContrast,
            input_kwargs={"Image": source, "Bright": 1.0, "Contrast": 4.0},
        )

    if glare:
        source = nw.new_node(
            Nodes.Glare,
            input_kwargs={"Image": source},
            attrs={"glare_type": "GHOSTS", "threshold": 0.5, "mix": -0.99},
        )

    if show:
        nw.new_node(Nodes.Composite, input_kwargs={"Image": source})

    return source.outputs[0] if hasattr(source, "outputs") else source


@gin.configurable
def configure_compositor_output(
    nw,
    frames_folder,
    image_denoised,
    image_noisy,
    passes_to_save,
    saving_ground_truth,
):
    file_output_node_png = nw.new_node(
        Nodes.OutputFile,
        attrs={
            "base_path": str(frames_folder),
            "format.file_format": "PNG",
            "format.color_mode": "RGB",
        },
    )
    file_output_node_exr = nw.new_node(
        Nodes.OutputFile,
        attrs={
            "base_path": str(frames_folder),
            "format.file_format": "OPEN_EXR",
            "format.color_mode": "RGB",
        },
    )
    default_file_output_node = (
        file_output_node_exr if saving_ground_truth else file_output_node_png
    )
    file_slot_list = []
    viewlayer = bpy.context.scene.view_layers["ViewLayer"]
    render_layers = nw.new_node(Nodes.RenderLayers)
    for viewlayer_pass, socket_name in passes_to_save:
        if hasattr(viewlayer, f"use_pass_{viewlayer_pass}"):
            setattr(viewlayer, f"use_pass_{viewlayer_pass}", True)
            # logger.info(f'viewlayer.use_pass_{viewlayer_pass}')
        else:
            setattr(viewlayer.cycles, f"use_pass_{viewlayer_pass}", True)
            # logger.info(f'viewlayer.cycles.use_pass_{viewlayer_pass}')

        # print all viewlayer attributes
        # logger.info('viewlayer attributes', dir(viewlayer))
        # logger.info('viewlayer.cycles attributes', dir(viewlayer.cycles))
        # must save the material pass index as EXR
        file_output_node = (
            default_file_output_node
            if viewlayer_pass != "material_index"
            else file_output_node_exr
        )

        logger.info(f"output keys: {render_layers.outputs.keys()}")

        slot_input = file_output_node.file_slots.new(socket_name)
        render_socket = render_layers.outputs[socket_name]
        match viewlayer_pass:
            case "vector":
                separate_color = nw.new_node(Nodes.CompSeparateColor, [render_socket])
                comnbine_color = nw.new_node(
                    Nodes.CompCombineColor,
                    [0, (separate_color, 3), (separate_color, 2), 0],
                )
                nw.links.new(comnbine_color.outputs[0], slot_input)
            case "normal" | "layered_normal":
                color = nw.new_node(
                    Nodes.CompositorMixRGB,
                    [None, render_socket, (0, 0, 0, 0)],
                    attrs={"blend_type": "ADD"},
                ).outputs[0]
                nw.links.new(color, slot_input)
            case _:
                nw.links.new(render_socket, slot_input)
        file_slot_list.append(file_output_node.file_slots[slot_input.name])

    slot_input = default_file_output_node.file_slots["Image"]
    image = image_denoised if image_denoised is not None else image_noisy
    nw.links.new(image, default_file_output_node.inputs["Image"])
    if saving_ground_truth:
        slot_input.path = "UniqueInstances"
    else:
        nw.links.new(image, file_output_node_exr.inputs["Image"])
        file_slot_list.append(file_output_node_exr.file_slots[slot_input.path])
    file_slot_list.append(default_file_output_node.file_slots[slot_input.path])

    return file_slot_list


def shader_emission(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler
    emission = nw.new_node(
        Nodes.Emission, input_kwargs={"Color": (1, 1, 1, 1), "Strength": 1}
    )

    nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": emission.outputs["Emission"]},
    )

def shader_refractive(nw: NodeWrangler, ior, roughness):
    refraction = nw.new_node(
        Nodes.RefractionBSDF, input_kwargs={"IOR": ior, "Roughness": 0}
    )

    nw.new_node(
        Nodes.MaterialOutput, input_kwargs={"Surface": refraction.outputs["BSDF"]}
    )


def get_ior_roughness(mat):
    nodes = mat.node_tree.nodes
    for node in nodes:
        if node.type == "BSDF_GLASS":
            return node.inputs["IOR"].default_value, node.inputs[
                "Roughness"
            ].default_value
    return 1.45, 0.0


def material_is_translucent(mat):
    for material in TRANSPARENT_MATERIAL_PREFIXES:
        if mat.name.split(".")[0] == material:
            return True

    # check if a node is refractive
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_REFRACTION":
            logger.info(f"Found refractive node in {mat.name}")
            return True
    return False


def global_flat_shading():
    for obj in bpy.context.scene.view_layers["ViewLayer"].objects:
        if "fire_system_type" in obj and obj["fire_system_type"] == "volume":
            continue
        if obj.name.lower() in {"atmosphere", "atmosphere_fine"}:
            bpy.data.objects.remove(obj)
        elif obj.active_material is not None:
            nw = obj.active_material.node_tree
            for node in nw.nodes:
                if node.bl_idname == Nodes.MaterialOutput:
                    vol_socket = node.inputs["Volume"]
                    if len(vol_socket.links) > 0:
                        nw.links.remove(vol_socket.links[0])
    bpy.context.view_layer.update()

    for obj in bpy.context.scene.view_layers["ViewLayer"].objects:
        if obj.type != "MESH":
            continue
        obj.hide_viewport = False
        if "fire_system_type" in obj and obj["fire_system_type"] == "gt_mesh":
            obj.hide_viewport = False
            obj.hide_render = False
        if not hasattr(obj, "material_slots"):
            print(obj.name, "NONE")
            continue

    for mat in bpy.data.materials:
        if material_is_translucent(mat):
            ior, roughness = get_ior_roughness(mat)
            mat.node_tree.nodes.clear()
            shader_refractive(NodeWrangler(mat.node_tree), ior, roughness)
            logger.info(f"Replaced {mat.name} with refractive shader")
        else:
            mat.node_tree.nodes.clear()
            nw = NodeWrangler(mat.node_tree)
            shader_emission(nw)
            logger.info(f"Replaced {mat.name} with random shader")

    for obj in bpy.context.scene.view_layers["ViewLayer"].objects:
        if len(obj.material_slots) == 0:
            surface.add_material(obj, shader_emission)

    nw = NodeWrangler(bpy.data.worlds["World"].node_tree)
    for link in nw.links:
        nw.links.remove(link)


def postprocess_blendergt_outputs(frames_folder, output_stem):
    # Save flow visualization
    flow_dst_path = frames_folder / f"Vector{output_stem}.exr"
    if flow_dst_path.is_file():
        flow_array = load_flow(flow_dst_path)
        np.save(flow_dst_path.with_name(f"Flow{output_stem}.npy"), flow_array)

        flow_color = colorize_flow(flow_array)
        if flow_color is not None:
            imwrite(
                flow_dst_path.with_name(f"Flow{output_stem}.png"),
                flow_color,
            )
        flow_dst_path.unlink()

    # Save surface normal visualization
    normal_dst_path = frames_folder / f"Normal{output_stem}.exr"
    if normal_dst_path.is_file():
        normal_array = load_normals(normal_dst_path)
        np.save(
            normal_dst_path.with_name(f"SurfaceNormal{output_stem}.npy"), normal_array
        )
        imwrite(
            normal_dst_path.with_name(f"SurfaceNormal{output_stem}.png"),
            colorize_normals(normal_array),
        )
        normal_dst_path.unlink()

    # Save depth visualization
    depth_dst_path = frames_folder / f"Depth{output_stem}.exr"
    if depth_dst_path.is_file():
        depth_array = load_depth(depth_dst_path)
        np.save(depth_dst_path.with_name(f"Depth{output_stem}.npy"), depth_array)
        imwrite(
            depth_dst_path.with_name(f"Depth{output_stem}.png"),
            colorize_depth(depth_array),
        )
        depth_dst_path.unlink()

    # Save segmentation visualization
    seg_dst_path = frames_folder / f"IndexOB{output_stem}.exr"
    if seg_dst_path.is_file():
        seg_mask_array = load_seg_mask(seg_dst_path)
        np.save(
            seg_dst_path.with_name(f"ObjectSegmentation{output_stem}.npy"),
            seg_mask_array,
        )
        imwrite(
            seg_dst_path.with_name(f"ObjectSegmentation{output_stem}.png"),
            colorize_int_array(seg_mask_array),
        )
        seg_dst_path.unlink()

    # Save unique instances visualization
    uniq_inst_path = frames_folder / f"UniqueInstances{output_stem}.exr"
    if uniq_inst_path.is_file():
        uniq_inst_array = load_uniq_inst(uniq_inst_path)
        np.save(
            uniq_inst_path.with_name(f"InstanceSegmentation{output_stem}.npy"),
            uniq_inst_array,
        )
        imwrite(
            uniq_inst_path.with_name(f"InstanceSegmentation{output_stem}.png"),
            colorize_int_array(uniq_inst_array),
        )
        uniq_inst_path.unlink()

    # Save Layered Depth Visualization
    for i in range(0, 8):
        depth_dst_path = frames_folder / f"Layered Depth {i}{output_stem}.exr"
        if depth_dst_path.is_file():
            depth_array = load_depth(depth_dst_path)
            np.save(
                depth_dst_path.with_name(f"Layered Depth {i}{output_stem}.npy"),
                depth_array,
            )
            imwrite(
                depth_dst_path.with_name(f"Layered Depth {i}{output_stem}.png"),
                colorize_depth(depth_array),
            )
            depth_dst_path.unlink()

    # Save Layered Normal Visualization
    for i in range(0, 8):
        normal_dst_path = frames_folder / f"Layered Normal {i}{output_stem}.exr"
        if normal_dst_path.is_file():
            normal_array = load_normals(normal_dst_path)
            np.save(
                normal_dst_path.with_name(f"Layered Normal {i}{output_stem}.npy"),
                normal_array,
            )
            imwrite(
                normal_dst_path.with_name(f"Layered Normal {i}{output_stem}.png"),
                colorize_normals(normal_array),
            )
            normal_dst_path.unlink()

    # Save Layered Object ID Visualization
    for i in range(0, 8):
        object_id_dst_path = frames_folder / f"Layered Object ID {i}{output_stem}.exr"
        if object_id_dst_path.is_file():
            object_id_array = load_seg_mask(object_id_dst_path)
            np.save(
                object_id_dst_path.with_name(f"Layered Object ID {i}{output_stem}.npy"),
                object_id_array,
            )
            imwrite(
                object_id_dst_path.with_name(f"Layered Object ID {i}{output_stem}.png"),
                colorize_int_array(object_id_array),
            )
            object_id_dst_path.unlink()

    # Save Layered Material ID Visualization
    for i in range(0, 8):
        material_id_dst_path = frames_folder / f"Layered Material ID {i}{output_stem}.exr"
        if material_id_dst_path.is_file():
            material_id_array = load_seg_mask(material_id_dst_path)
            np.save(
                material_id_dst_path.with_name(
                    f"Layered Material ID {i}{output_stem}.npy"
                ),
                material_id_array,
            )
            imwrite(
                material_id_dst_path.with_name(
                    f"Layered Material ID {i}{output_stem}.png"
                ),
                colorize_int_array(material_id_array),
            )
            material_id_dst_path.unlink()


def postprocess_materialgt_output(frames_folder, output_stem):
    # Save material segmentation visualization if present
    ma_seg_dst_path = frames_folder / f"IndexMA{output_stem}.exr"
    if ma_seg_dst_path.is_file():
        ma_seg_mask_array = load_seg_mask(ma_seg_dst_path)
        np.save(
            ma_seg_dst_path.with_name(f"MaterialSegmentation{output_stem}.npy"),
            ma_seg_mask_array,
        )
        imwrite(
            ma_seg_dst_path.with_name(f"MaterialSegmentation{output_stem}.png"),
            colorize_int_array(ma_seg_mask_array),
        )
        ma_seg_dst_path.unlink()


def configure_compositor(
    frames_folder: Path,
    passes_to_save: list,
    flat_shading: bool,
):
    compositor_node_tree = bpy.context.scene.node_tree
    nw = NodeWrangler(compositor_node_tree)

    render_layers = nw.new_node(Nodes.RenderLayers)
    final_image_denoised = compositor_postprocessing(
        nw, source=render_layers.outputs["Image"]
    )

    final_image_noisy = (
        compositor_postprocessing(
            nw, source=render_layers.outputs["Noisy Image"], show=False
        )
        if bpy.context.scene.cycles.use_denoising
        else None
    )

    return configure_compositor_output(
        nw,
        frames_folder,
        image_denoised=final_image_denoised,
        image_noisy=final_image_noisy,
        passes_to_save=passes_to_save,
        saving_ground_truth=flat_shading,
    )

def get_camera_shifts(shifted_camera_num):
    render = bpy.context.scene.render
    camera_shifts_list = [
        [0, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]
    ]
    shift_amount = 0.1
    shift_x = shift_amount / render.resolution_x
    shift_y = shift_amount / render.resolution_y
    return [[camera_shifts_list[i][0] * shift_x, camera_shifts_list[i][1] * shift_y] for i in range(shifted_camera_num)]

def aggregate_shifted_renders(
    frames_folder,
    indices,
    shifted_camera_num,
    consistency_m=0.1,
):
    """Aggregate subpixel shifts and keep depth-consistency diagnostics."""
    aggregate_stem = get_suffix(dict(**indices, resample=0))
    for i in range(0, 8):
        layered_depths = []
        for shift_index in range(shifted_camera_num):
            shift_stem = get_suffix(dict(**indices, resample=shift_index))
            layered_depth_path = frames_folder / f"Layered Depth {i}{shift_stem}.npy"
            layered_depth = np.load(layered_depth_path)
            layered_depths.append(layered_depth)
        layered_depths = np.array(layered_depths)
        valid = (
            np.isfinite(layered_depths)
            & (layered_depths > 0)
            & (layered_depths < 1e9)
        )
        safe = np.where(valid, layered_depths, 0)
        spread = safe.max(axis=0) - safe.min(axis=0)
        stable = valid.all(axis=0) & (spread <= float(consistency_m))
        layered_depth = np.where(stable, safe.mean(axis=0), 0).astype(np.float32)
        np.save(frames_folder / f"Layered Depth {i}{aggregate_stem}.npy", layered_depth)
        imwrite(frames_folder / f"Layered Depth {i}{aggregate_stem}.png", colorize_depth(layered_depth))
        np.save(
            frames_folder / f"Layered Depth {i} Consistency{aggregate_stem}.npy",
            stable.astype(np.uint8),
        )
        np.save(
            frames_folder / f"Layered Depth {i} ValidShiftFraction{aggregate_stem}.npy",
            valid.mean(axis=0).astype(np.float32),
        )
        np.save(
            frames_folder / f"Layered Depth {i} ShiftRange{aggregate_stem}.npy",
            spread.astype(np.float32),
        )

        layered_object_ids = []
        for shift_index in range(shifted_camera_num):
            shift_stem = get_suffix(dict(**indices, resample=shift_index))
            layered_object_id_path = frames_folder / f"Layered Object ID {i}{shift_stem}.npy"
            if layered_object_id_path.is_file():
                layered_object_ids.append(np.load(layered_object_id_path))
        if len(layered_object_ids) != shifted_camera_num:
            logger.warning(
                f"Skipping object-ID aggregation for layer {i}: "
                "one or more shifted passes are missing"
            )
        else:
            layered_object_ids = np.array(layered_object_ids)
            object_stable = (
                layered_object_ids.max(axis=0) - layered_object_ids.min(axis=0)
            ) <= 0
            layered_object_id = np.where(
                object_stable,
                layered_object_ids.mean(axis=0),
                0,
            )
            np.save(frames_folder / f"Layered Object ID {i}{aggregate_stem}.npy", layered_object_id)
            imwrite(
                frames_folder / f"Layered Object ID {i}{aggregate_stem}.png",
                colorize_int_array(layered_object_id),
            )

        layered_normals = []
        for shift_index in range(shifted_camera_num):
            shift_stem = get_suffix(dict(**indices, resample=shift_index))
            normal_path = frames_folder / f"Layered Normal {i}{shift_stem}.npy"
            if normal_path.is_file():
                layered_normals.append(np.load(normal_path))
        if len(layered_normals) == shifted_camera_num:
            layered_normals = np.asarray(layered_normals, dtype=np.float32)
            stable_slice = stable.astype(np.float32)[..., None]
            layered_normal = (
                layered_normals * stable_slice
            ).sum(axis=0) / np.maximum(stable_slice.sum(axis=0), 1.0)
            norm = np.linalg.norm(layered_normal, axis=-1, keepdims=True)
            layered_normal = np.where(norm > 1e-8, layered_normal / np.maximum(norm, 1e-8), 0)
            np.save(frames_folder / f"Layered Normal {i}{aggregate_stem}.npy", layered_normal)
            imwrite(
                frames_folder / f"Layered Normal {i}{aggregate_stem}.png",
                colorize_normals(layered_normal),
            )
        elif layered_normals:
            logger.warning(
                f"Skipping normal aggregation for layer {i}: "
                "shifted normal pass count is incomplete"
            )


@gin.configurable
def merge_duplicate_layers(
    frames_folder,
    indices,
    shifted_camera_num,
    merge_tol=0.03,
    merge_same_object=True,
):
    """Collapse the two interfaces of a closed transparent shell into one layer.

    A closed glass shell (window pane, glass cabinet door, jar, aquarium wall) is hit
    twice by the same ray, so it occupies two layer slots a couple of centimeters
    apart, e.g. layer0 = outer face, layer1 = inner face, layer2 = whatever is behind.
    For the LayeredDepth task the meaningful stack is one entry per *distinct* surface,
    so drop any layer that is within `merge_tol` of the layer above it (and, when object
    ids are available, belongs to the same object) and shift the rest up.
    """

    def valid(depth):
        # The world background renders as a very large depth.
        return (depth > 0) & (depth < 1e9)

    for shift_index in range(shifted_camera_num):
        suffix = get_suffix(dict(**indices, resample=shift_index))
        passes = {}
        for kind in ("Layered Depth", "Layered Object ID", "Layered Normal"):
            paths = [frames_folder / f"{kind} {i}{suffix}.npy" for i in range(0, 8)]
            if all(p.is_file() for p in paths):
                passes[kind] = (paths, np.stack([np.load(p) for p in paths], axis=0))
        if "Layered Depth" not in passes:
            continue

        depths = passes["Layered Depth"][1]
        object_ids = passes.get("Layered Object ID")
        object_ids = object_ids[1] if object_ids is not None else None
        valid_depth = valid(depths)

        keep = np.zeros(depths.shape, dtype=bool)
        last_depth = np.zeros(depths.shape[1:], dtype=np.float32)
        last_object = np.zeros(depths.shape[1:], dtype=np.int64)
        has_last = np.zeros(depths.shape[1:], dtype=bool)
        for i in range(depths.shape[0]):
            close = (
                valid_depth[i]
                & has_last
                & (np.abs(depths[i] - last_depth) < merge_tol)
            )
            if merge_same_object and object_ids is not None:
                ids = np.rint(object_ids[i]).astype(np.int64)
                close &= (ids == last_object) | (ids <= 0) | (last_object <= 0)
            keep[i] = valid_depth[i] & ~close
            last_depth = np.where(keep[i], depths[i], last_depth)
            if object_ids is not None:
                last_object = np.where(
                    keep[i], np.rint(object_ids[i]).astype(np.int64), last_object
                )
            has_last |= keep[i]

        dropped = int((valid_depth & ~keep).sum())
        if dropped == 0:
            continue

        # Compact every pass with the same keep mask so the layers stay consistent.
        layer_index, row, col = np.nonzero(keep)
        rank = np.cumsum(keep, axis=0)[layer_index, row, col] - 1

        def compact(stack):
            out = np.zeros_like(stack)
            out[rank, row, col] = stack[layer_index, row, col]
            return out

        merged_depth = compact(depths)
        for i in range(8):
            np.save(passes["Layered Depth"][0][i], merged_depth[i])
            imwrite(
                passes["Layered Depth"][0][i].with_suffix(".png"),
                colorize_depth(merged_depth[i]),
            )

        if object_ids is not None:
            merged_object_id = compact(object_ids)
            for i in range(8):
                np.save(passes["Layered Object ID"][0][i], merged_object_id[i])
                imwrite(
                    passes["Layered Object ID"][0][i].with_suffix(".png"),
                    colorize_int_array(merged_object_id[i]),
                )

        if "Layered Normal" in passes:
            merged_normal = compact(passes["Layered Normal"][1])
            for i in range(8):
                np.save(passes["Layered Normal"][0][i], merged_normal[i])
                imwrite(
                    passes["Layered Normal"][0][i].with_suffix(".png"),
                    colorize_normals(merged_normal[i]),
                )

        for diagnostic in (
            "Consistency",
            "ValidShiftFraction",
            "ShiftRange",
        ):
            diagnostic_path = (
                frames_folder
                / f"Layered Depth 0 {diagnostic}{suffix}.npy"
            )
            if diagnostic_path.is_file():
                diagnostic_stack = np.stack(
                    [
                        np.load(
                            frames_folder
                            / f"Layered Depth {i} {diagnostic}{suffix}.npy"
                        )
                        for i in range(8)
                    ],
                    axis=0,
                )
                merged_diagnostic = compact(diagnostic_stack)
                for i in range(8):
                    np.save(
                        frames_folder
                        / f"Layered Depth {i} {diagnostic}{suffix}.npy",
                        merged_diagnostic[i],
                    )

        logger.info(
            f"Merged {dropped} duplicate glass-shell layers "
            f"({100.0 * dropped / max(int(valid_depth.sum()), 1):.2f}% of valid layer pixels)"
        )


@gin.configurable
def render_image(
    camera: bpy.types.Object,
    frames_folder,
    passes_to_save,
    render_resolution_override=None,
    excludes=[],
    use_dof=False,
    dof_aperture_fstop=2.8,
    flat_shading=False,
    override_num_samples=None,
    shifted_camera_num=2,
    merge_duplicate_glass_layers=True,
    merge_duplicate_tol=0.03,
    merge_duplicate_same_object=True,
    gt_consistency_m=0.1,
):
    tic = time.time()
    if not flat_shading:
        shifted_camera_num = 1

    for exclude in excludes:
        bpy.data.objects[exclude].hide_render = True

    init.configure_cycles_devices()
    for material in bpy.data.materials:
        set_geometry_option(material)

    tmp_dir = frames_folder.parent.resolve() / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    bpy.context.scene.render.filepath = f"{tmp_dir}{os.sep}"

    camrig_id, subcam_id = cam_util.get_id(camera)

    if override_num_samples is not None:  # usually used for GT
        bpy.context.scene.cycles.samples = override_num_samples

    if flat_shading:
        with Timer("Set object indices"):
            object_data = set_pass_indices()
            json_object = json.dumps(object_data, indent=4)
            first_frame = bpy.context.scene.frame_start
            suffix = get_suffix(
                dict(
                    cam_rig=camrig_id,
                    resample=0,
                    frame=first_frame,
                    subcam=subcam_id,
                )
            )
            (frames_folder / f"Objects{suffix}.json").write_text(json_object)

        if "layered_material_id" in (x[0] for x in passes_to_save):
            # Flat shading keeps the original material datablocks (only their node trees
            # are replaced), so material pass indices still identify the real materials.
            with Timer("Set material indices"):
                material_data = set_material_pass_indices()
                json_object = json.dumps(material_data, indent=4)
                first_frame = bpy.context.scene.frame_start
                suffix = get_suffix(
                    dict(
                        cam_rig=camrig_id,
                        resample=0,
                        frame=first_frame,
                        subcam=subcam_id,
                    )
                )
                (frames_folder / f"Materials{suffix}.json").write_text(json_object)

        with Timer("Flat Shading"):
            global_flat_shading()
    else:
        segment_materials = "material_index" in (x[0] for x in passes_to_save)
        if segment_materials:
            with Timer("Set material indices"):
                material_data = set_material_pass_indices()
                json_object = json.dumps(material_data, indent=4)
                first_frame = bpy.context.scene.frame_start
                suffix = get_suffix(
                    dict(
                        cam_rig=camrig_id,
                        resample=0,
                        frame=first_frame,
                        subcam=subcam_id,
                    )
                )
                (frames_folder / f"Materials{suffix}.json").write_text(json_object)

    if not bpy.context.scene.use_nodes:
        bpy.context.scene.use_nodes = True
    file_slot_nodes = configure_compositor(frames_folder, passes_to_save, flat_shading)

    if use_dof == "IF_TARGET_SET":
        use_dof = camera.data.dof.focus_object is not None
    elif use_dof is not None:
        camera.data.dof.use_dof = use_dof
        camera.data.dof.aperture_fstop = dof_aperture_fstop

    if render_resolution_override is not None:
        bpy.context.scene.render.resolution_x = render_resolution_override[0]
        bpy.context.scene.render.resolution_y = render_resolution_override[1]

    camera_shifts = get_camera_shifts(shifted_camera_num)

    original_paths = {}
    for file_slot in file_slot_nodes:
        original_paths[file_slot] = str(file_slot.path)
        logger.info(f"Original path: {original_paths[file_slot]}")

    for camera_shift_index, camera_shift in enumerate(camera_shifts):
        camera.data.shift_x += camera_shift[0]
        camera.data.shift_y += camera_shift[1]

        indices = dict(cam_rig=camrig_id, resample=camera_shift_index, subcam=subcam_id)

        ## Update output names
        fileslot_suffix = get_suffix({"frame": "####", **indices})
        for file_slot in file_slot_nodes:
            file_slot.path = f"{original_paths[file_slot]}{fileslot_suffix}"
            logger.info(f"File slot path: {file_slot.path}")
        

        # Render the scene
        bpy.context.scene.camera = camera
        with Timer("Actual rendering"):
            bpy.ops.render.render(animation=True)

        with Timer("Post Processing"):
            for frame in range(
                bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1
            ):
                if flat_shading:
                    bpy.context.scene.frame_set(frame)
                    suffix = get_suffix(dict(frame=frame, **indices))
                    postprocess_blendergt_outputs(frames_folder, suffix)
                else:
                    cam_util.save_camera_parameters(
                        camera,
                        output_folder=frames_folder,
                        frame=frame,
                    )
                    bpy.context.scene.frame_set(frame)
                    suffix = get_suffix(dict(frame=frame, **indices))
                    postprocess_materialgt_output(frames_folder, suffix)

    if shifted_camera_num > 1:
        for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
            indices = dict(cam_rig=camrig_id, subcam=subcam_id, frame=frame)
            aggregate_shifted_renders(
                frames_folder,
                indices,
                shifted_camera_num,
                consistency_m=gt_consistency_m,
            )

    if flat_shading and merge_duplicate_glass_layers:
        # Layer slots are bounce indices, so a closed glass shell occupies two of them.
        # Collapse those pairs so layer k means "the k-th distinct surface".
        for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
            indices = dict(cam_rig=camrig_id, subcam=subcam_id, frame=frame)
            merge_duplicate_layers(
                frames_folder,
                indices,
                shifted_camera_num,
                merge_tol=merge_duplicate_tol,
                merge_same_object=merge_duplicate_same_object,
            )

    for file in tmp_dir.glob("*.png"):
        file.unlink()

    reorganize_old_framesfolder(frames_folder)

    logger.info(f"rendering time: {time.time() - tic}")


# ---------------------------------------------------------------------------
# Depth-from-focus (DFF) focal stacks
# ---------------------------------------------------------------------------


def _default_layered_passes():
    """Same layer set as flat/render_image.passes_to_save in configs_nature/base.gin."""
    passes = [
        ["z", "Depth"],
        ["normal", "Normal"],
        ["object_index", "IndexOB"],
    ]
    for kind, label in (
        ("layered_depth", "Layered Depth"),
        ("layered_normal", "Layered Normal"),
        ("layered_object_id", "Layered Object ID"),
    ):
        passes += [[kind, f"{label} {i}"] for i in range(8)]
    return passes


def _camera_visible_depths(camera, grid=(96, 54)):
    """Camera-Z depth of the first surface for a grid of camera rays.

    Uses scene ray casts instead of a render pass: it needs no material changes (so the
    realistic DOF renders can follow) and no extra render of the scene.
    """
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mw = np.array(camera.matrix_world, dtype=np.float64)
    rot = mw[:3, :3]
    origin = mw[:3, 3]
    forward = rot @ np.array([0.0, 0.0, -1.0])

    corners = [np.array(v[:], dtype=np.float64) for v in camera.data.view_frame(scene=scene)]
    top_right, bottom_right, bottom_left, top_left = corners
    widths, heights = grid
    us = np.linspace(0.0, 1.0, widths)
    vs = np.linspace(0.0, 1.0, heights)
    depths = []
    with _raycast_visible_assets():
        for v in vs:
            left = top_left + (bottom_left - top_left) * v
            right = top_right + (bottom_right - top_right) * v
            for p in left + (right - left) * us[:, None]:
                direction = rot @ p
                norm = np.linalg.norm(direction)
                if norm < 1e-9:
                    continue
                hit, loc, *_ = scene.ray_cast(
                    depsgraph,
                    tuple(origin),
                    tuple(direction / norm),
                )
                if hit:
                    depths.append(float((np.array(loc) - origin) @ forward))
    return np.array([d for d in depths if d > 1e-3])


def _reference_depths(folder):
    """First-layer depths from an existing Blender layered-GT folder."""
    folder = Path(folder)
    files = sorted(folder.rglob("Layered Depth 0_*.npy"))
    if not files:
        raise FileNotFoundError(f"no 'Layered Depth 0_*.npy' under {folder}")
    samples = []
    for path in files:
        d = np.load(path).ravel()
        samples.append(d[(d > 0) & (d < 1e9)])
    return np.concatenate(samples)


def _focus_planes(depths, count, spacing="diopters", percentiles=(2.0, 98.0), margin=0.1):
    lo, hi = np.percentile(depths, percentiles)
    lo = max(0.05, float(lo) * (1.0 - margin))
    hi = float(hi) * (1.0 + margin)
    if hi <= lo:
        hi = lo * 1.5
    if spacing == "diopters":
        # Uniform in 1/z, which is how defocus blur is distributed.
        inv = np.linspace(1.0 / lo, 1.0 / hi, count)
        return [float(1.0 / d) for d in inv], (lo, hi)
    return [float(z) for z in np.linspace(lo, hi, count)], (lo, hi)


def _blur_px(focus_distance, depths, lens_mm, fstop, sensor_width_mm, res_x):
    """Thin-lens circle of confusion, in pixels, for each sampled depth."""
    f = lens_mm / 1000.0
    d = np.asarray(depths, dtype=np.float64)
    coc_m = (f**2 / (fstop * (focus_distance - f))) * np.abs(d - focus_distance) / d
    return coc_m / (sensor_width_mm / 1000.0) * res_x


def _camera_calibration(camera):
    """Return K and camera transforms in a portable JSON representation."""
    from mathutils import Matrix

    scene = bpy.context.scene
    render = scene.render
    width = int(render.resolution_x * render.resolution_percentage / 100)
    height = int(render.resolution_y * render.resolution_percentage / 100)
    projection = camera.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(),
        x=width,
        y=height,
        scale_x=render.pixel_aspect_x,
        scale_y=render.pixel_aspect_y,
    )
    to_cv = Matrix(
        ((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1))
    )
    return {
        "available": True,
        "name": camera.name,
        "resolution": [width, height],
        "frame": scene.frame_current,
        "K": [
            [
                projection[0][0] * width / 2,
                0,
                (1 - projection[0][2]) * width / 2,
            ],
            [
                0,
                projection[1][1] * height / 2,
                (1 + projection[1][2]) * height / 2,
            ],
            [0, 0, 1],
        ],
        "world_to_camera_cv": [
            list(row)
            for row in (
                to_cv @ camera.matrix_world.normalized().inverted()
            )
        ],
        "camera_to_world_blender": [
            list(row) for row in camera.matrix_world
        ],
        "coordinates": "RGB top-left origin; CV camera x-right, y-down, z-forward",
        "world_unit_to_meters": scene.unit_settings.scale_length,
        "lens_mm": camera.data.lens,
        "sensor_width_mm": camera.data.sensor_width,
        "sensor_height_mm": camera.data.sensor_height,
        "sensor_fit": camera.data.sensor_fit,
        "shift_x": camera.data.shift_x,
        "shift_y": camera.data.shift_y,
        "pixel_aspect": [
            render.pixel_aspect_x,
            render.pixel_aspect_y,
        ],
        "dof_enabled": camera.data.dof.use_dof,
        "aperture_fstop": camera.data.dof.aperture_fstop,
        "focus_distance_scene_units": camera.data.dof.focus_distance,
        "view_transform": scene.view_settings.view_transform,
        "exposure_ev": scene.view_settings.exposure,
    }


def _apply_external_view_config(scene, config_path):
    """Create an independent camera from a planned view JSON file."""
    from mathutils import Matrix

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    source = scene.camera
    if source is None:
        raise RuntimeError("external view config requires an active source camera")
    requested_name = config.get("camera_name", "camera_0_0")
    match = re.fullmatch(r"camera_(\d+)_(\d+)", requested_name)
    if match:
        camera_rig = int(match.group(1))
        subcam = int(match.group(2))
        while bpy.data.objects.get(f"camera_{camera_rig}_{subcam}") is not None:
            camera_rig += 1000
        requested_name = f"camera_{camera_rig}_{subcam}"
    camera = bpy.data.objects.new(
        requested_name,
        source.data.copy(),
    )
    scene.collection.objects.link(camera)
    camera.matrix_world = Matrix(config["camera_to_world_blender"])
    for name, value in config.get("camera_settings", {}).items():
        if name in {"rig", "subcam"}:
            continue
        if hasattr(camera.data, name):
            setattr(camera.data, name, value)
    camera.data.dof.focus_object = None
    scene.camera = camera
    bpy.context.view_layer.update()
    return camera, config


def _camera_glass_stats(camera, grid=(72, 40)):
    """Ray-cast a camera view and report transparent coverage and object hits."""
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mw = np.array(camera.matrix_world, dtype=np.float64)
    rot, origin = mw[:3, :3], mw[:3, 3]
    corners = [np.array(v[:], dtype=np.float64) for v in camera.data.view_frame(scene=scene)]
    top_right, bottom_right, bottom_left, top_left = corners
    hits = total = 0
    hit_names = {}
    transparent_hits = {}
    with _raycast_visible_assets():
        for v in np.linspace(0.0, 1.0, grid[1]):
            left = top_left + (bottom_left - top_left) * v
            right = top_right + (bottom_right - top_right) * v
            for p in left + (right - left) * np.linspace(0.0, 1.0, grid[0])[:, None]:
                direction = rot @ p
                norm = np.linalg.norm(direction)
                if norm < 1e-9:
                    continue
                total += 1
                hit, _, _, _, obj, _ = scene.ray_cast(
                    depsgraph, tuple(origin), tuple(direction / norm)
                )
                if not hit or obj is None or obj.type != "MESH":
                    continue
                hit_names[obj.name] = hit_names.get(obj.name, 0) + 1
                names = [m.name for m in obj.data.materials if m is not None]
                if (
                    not _is_architectural_glass(obj)
                    and any(n.startswith(TRANSPARENT_MATERIAL_PREFIXES) for n in names)
                ):
                    hits += 1
                    transparent_hits[obj.name] = transparent_hits.get(obj.name, 0) + 1
    coverage = hits / max(total, 1)
    top = sorted(hit_names.items(), key=lambda kv: -kv[1])[:3]
    return coverage, transparent_hits, total, top


def _camera_glass_coverage(camera, grid=(72, 40)):
    """Fraction of camera rays whose first hit is a transparent object."""
    coverage, _, total, top = _camera_glass_stats(camera, grid=grid)
    logger.info(
        f"glass coverage raycast: {total} rays, top hits "
        f"{[(n.split('(')[0][:24], c) for n, c in top]}, "
        f"camera={camera.name} lens={camera.data.lens:.1f}"
    )
    return coverage


def _load_glass_target_usage(frames_folder):
    usage = {}
    for path in Path(frames_folder).glob("focal_stack_camera_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        target = (data.get("aimed_at") or {}).get("object")
        if target:
            usage[target] = usage.get(target, 0) + 1
    return usage


def _aim_camera_at_glass(
    camera,
    target_usage=None,
    diversity_weight=0.05,
    max_targets=16,
    score_grid=(32, 18),
):
    """Rotate the camera to a view covering several distinct transparent objects."""
    from mathutils import Vector

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cam_loc = camera.matrix_world.translation
    original_rotation = camera.rotation_euler.copy()
    target_usage = target_usage or {}
    candidates = []
    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        if _is_architectural_glass(obj):
            continue
        names = [m.name for m in obj.data.materials if m is not None]
        if not any(n.startswith(TRANSPARENT_MATERIAL_PREFIXES) for n in names):
            continue
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        center = sum(corners, Vector()) / len(corners)
        lo = Vector(
            (
                min(c.x for c in corners),
                min(c.y for c in corners),
                min(c.z for c in corners),
            )
        )
        hi = Vector(
            (
                max(c.x for c in corners),
                max(c.y for c in corners),
                max(c.z for c in corners),
            )
        )
        size = hi - lo
        volume = max(size.x, 1e-3) * max(size.y, 1e-3) * max(size.z, 1e-3)
        distance = (center - cam_loc).length
        if distance < 0.2 or volume < 1e-6:
            continue
        score = volume / (distance**2) * (10 ** _glass_priority(obj))
        candidates.append((score, obj, center, distance))

    best = None
    evaluated = 0
    with _raycast_visible_assets():
        for score, obj, center, distance in sorted(candidates, key=lambda c: -c[0]):
            if evaluated >= int(max_targets):
                break
            direction = center - cam_loc
            hit, _, _, _, hit_obj, _ = scene.ray_cast(
                depsgraph,
                tuple(cam_loc),
                tuple(direction.normalized()),
                distance=distance * 1.01,
            )
            if not hit or hit_obj != obj:
                continue

            evaluated += 1
            world_rot = direction.to_track_quat("-Z", "Y")
            if camera.parent is not None:
                parent_rot = camera.parent.matrix_world.to_quaternion()
                world_rot = parent_rot.inverted() @ world_rot
            camera.rotation_mode = "XYZ"
            camera.rotation_euler = world_rot.to_euler()
            bpy.context.view_layer.update()
            coverage, transparent_hits, _, _ = _camera_glass_stats(
                camera, grid=score_grid
            )
            reuse = int(target_usage.get(obj.name, 0))
            view_score = (
                coverage
                + 0.08 * len(transparent_hits)
                - float(diversity_weight) * reuse
            )
            if best is None or view_score > best["score"]:
                best = {
                    "score": view_score,
                    "name": obj.name,
                    "distance": distance,
                    "rotation": world_rot.to_euler(),
                    "coverage": coverage,
                    "objects": sorted(transparent_hits),
                }

    if best is not None:
        camera.rotation_mode = "XYZ"
        camera.rotation_euler = best["rotation"]
        bpy.context.view_layer.update()
        return (
            best["name"],
            best["distance"],
            {
                "coverage": best["coverage"],
                "objects": best["objects"],
            },
        )

    camera.rotation_euler = original_rotation
    bpy.context.view_layer.update()
    return None


def _add_glass_fill_light(target, camera, energy, size, distance):
    """Add a temporary softbox on the camera side of the target glass object."""
    from mathutils import Vector

    corners = [target.matrix_world @ Vector(c) for c in target.bound_box]
    center = sum(corners, Vector()) / len(corners)
    cam_loc = camera.matrix_world.translation
    toward_cam = cam_loc - center
    if toward_cam.length < 1e-6:
        toward_cam = Vector((0.0, 0.0, 1.0))
    loc = center + toward_cam.normalized() * float(distance) + Vector((0.0, 0.0, 0.35))

    light_data = bpy.data.lights.new("dff_fill_light", type="AREA")
    light_data.energy = float(energy)
    light_data.shape = "DISK"
    light_data.size = float(size)
    light_obj = bpy.data.objects.new("dff_fill_light", light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.location = loc
    light_obj.rotation_euler = (center - loc).to_track_quat("-Z", "Y").to_euler()
    return light_obj


@gin.configurable
def render_focal_stack(
    camera: bpy.types.Object,
    frames_folder,
    passes_to_save=None,
    gt_passes_to_save=None,
    focus_count=12,
    aperture_fstop=2.0,
    aperture_blades=6,
    focus_spacing="diopters",
    depth_percentiles=(2.0, 98.0),
    focus_margin=0.1,
    focus_distances=None,
    depth_reference_folder=None,
    depth_sample_grid=(96, 54),
    focal_length_mm=None,
    rgb_samples=None,
    resolution_override=None,
    fill_light_energy=0.0,
    fill_light_size=1.5,
    fill_light_distance=1.2,
    save_gt=True,
    gt_shifted_camera_num=5,
    gt_consistency_m=0.1,
    merge_duplicate_glass_layers=True,
    view_config_path=None,
    write_metadata=True,
    aim_at_glass=False,
    aim_diversity=True,
    min_glass_coverage=None,
    min_glass_objects=None,
):
    """Render a DFF focal stack: N images, same pose, different focus distance.

    Layout (all paths relative to `frames_folder`):
        focal_stack.json            focus distances, aperture, depth range, blur stats
        focus_00/frames/Image/...   one image per focus distance
        ...
        gt/frames/Layered Depth k/  layered depth GT (focus independent, rendered last
                                    because flat shading replaces the materials)

    The focus distances are chosen from the *visible* depth range (percentiles of the
    first-hit depth), uniformly spaced in 1/z (diopters), which is how defocus blur is
    distributed. Pass `focus_distances=[...]` to override the planning.
    """
    frames_folder = Path(frames_folder)
    frames_folder.mkdir(parents=True, exist_ok=True)
    if passes_to_save is None:
        passes_to_save = [["z", "Depth"]]
    if gt_passes_to_save is None:
        gt_passes_to_save = _default_layered_passes()

    scene = bpy.context.scene
    applied_view_config = None
    if view_config_path is None:
        view_config_path = os.environ.get("DFF_VIEW_CONFIG")
    if view_config_path:
        camera, applied_view_config = _apply_external_view_config(
            scene, view_config_path
        )
        if aim_at_glass:
            logger.warning(
                "external view config already defines the camera pose; "
                "ignoring aim_at_glass"
            )
            aim_at_glass = False
        logger.info(
            f"applied external view config {view_config_path!r} "
            f"as {camera.name}"
        )
    if resolution_override is not None:
        scene.render.resolution_x, scene.render.resolution_y = resolution_override
    res_x, res_y = scene.render.resolution_x, scene.render.resolution_y

    cam = camera.data
    # Apply the DFF lens before measuring coverage / planning focus: the lens determines
    # what is in frame.
    if focal_length_mm is not None:
        cam.lens = float(focal_length_mm)
        bpy.context.view_layer.update()
    aimed_at = None
    if aim_at_glass:
        target_usage = _load_glass_target_usage(frames_folder) if aim_diversity else {}
        aim = _aim_camera_at_glass(camera, target_usage=target_usage)
        if aim is None:
            logger.warning("aim_at_glass: no transparent object found, keeping camera pose")
        else:
            aimed_at = {
                "object": aim[0],
                "distance_m": round(float(aim[1]), 3),
                "glass_coverage": round(float(aim[2]["coverage"]), 4),
                "glass_objects_visible": aim[2]["objects"],
            }
            bpy.context.view_layer.update()
            logger.info(
                f"aim_at_glass: camera now looks at {aim[0]!r} ({aim[1]:.2f} m away), "
                f"{len(aim[2]['objects'])} transparent objects visible"
            )
    glass_coverage, glass_hits, _, _ = _camera_glass_stats(camera)
    glass_objects = sorted(glass_hits)
    logger.info(
        f"camera {camera.name}: first-hit glass coverage "
        f"{100 * glass_coverage:.1f}%, objects={len(glass_objects)}"
    )
    coverage_ok = (
        min_glass_coverage is None
        or glass_coverage >= float(min_glass_coverage)
    )
    objects_ok = (
        min_glass_objects is None
        or len(glass_objects) >= int(min_glass_objects)
    )
    if not coverage_ok or not objects_ok:
        logger.warning(
            f"skipping focal stack for {camera.name}: coverage={glass_coverage:.3f}, "
            f"objects={len(glass_objects)}"
        )
        if write_metadata:
            (frames_folder / f"focal_stack_{camera.name}.json").write_text(
                json.dumps(
                    {
                        "kind": "dff_focal_stack",
                        "camera": camera.name,
                        "skipped": True,
                        "glass_coverage": round(glass_coverage, 4),
                        "glass_objects_visible": glass_objects,
                        "min_glass_coverage": min_glass_coverage,
                        "min_glass_objects": min_glass_objects,
                    },
                    indent=2,
                )
            )
        return {}
    if fill_light_energy > 0 and aimed_at is not None:
        target_obj = bpy.data.objects.get(aimed_at["object"])
        if target_obj is not None:
            _add_glass_fill_light(
                target_obj,
                camera,
                energy=fill_light_energy,
                size=fill_light_size,
                distance=fill_light_distance,
            )
            logger.info(
                f"added DFF fill light: {fill_light_energy:g} W area, "
                f"size {fill_light_size:g} m"
            )
    camera_rig_id, camera_subcam_id = cam_util.get_id(camera)
    cam.dof.use_dof = True
    cam.dof.focus_object = None
    cam.dof.aperture_fstop = float(aperture_fstop)
    if aperture_blades:
        cam.dof.aperture_blades = int(aperture_blades)
    lens_mm = float(cam.lens)
    sensor_width_mm = float(cam.sensor_width)
    fstop = float(cam.dof.aperture_fstop)
    camera_calibration = _camera_calibration(camera)

    if depth_reference_folder is not None:
        depths = _reference_depths(depth_reference_folder)
        depth_source = f"gt:{depth_reference_folder}"
    else:
        depths = _camera_visible_depths(camera, grid=depth_sample_grid)
        depth_source = "raycast"
    if depths.size < 16:
        raise RuntimeError(f"not enough visible depth samples ({depths.size}) to plan focus planes")

    explicit_focus = focus_distances is not None
    if not explicit_focus:
        focus_distances, depth_window = _focus_planes(
            depths,
            focus_count,
            spacing=focus_spacing,
            percentiles=depth_percentiles,
            margin=focus_margin,
        )
    else:
        focus_distances = [float(z) for z in focus_distances]
        depth_window = (float(np.min(depths)), float(np.max(depths)))

    planes = []
    for z in focus_distances:
        blur = _blur_px(z, depths, lens_mm, fstop, sensor_width_mm, res_x)
        planes.append(
            {
                "focus_distance_m": round(z, 4),
                "blur_px_median": round(float(np.median(blur)), 3),
                "blur_px_max": round(float(np.max(blur)), 3),
            }
        )
    max_blur = max(p["blur_px_max"] for p in planes)
    if max_blur < 1.5:
        logger.warning(
            f"DFF stack has weak defocus (max blur {max_blur:.2f} px): "
            f"raise focal_length_mm ({lens_mm:.1f}) / open the aperture (f/{fstop:g}) "
            f"or widen the focus range."
        )

    logger.info(
        f"DFF focal stack: {len(planes)} planes, depth window "
        f"{depth_window[0]:.2f}-{depth_window[1]:.2f} m ({depth_source}), "
        f"lens {lens_mm:.1f} mm, f/{fstop:g}, max blur {max_blur:.2f} px"
    )

    # RGB renders first: flat shading below irreversibly replaces the materials.
    camera.data.shift_x = 0.0
    camera.data.shift_y = 0.0
    for i, z in enumerate(focus_distances):
        frames = frames_folder / f"focus_{i:02d}" / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        camera.data.dof.focus_distance = z
        logger.info(f"DFF focus {i + 1}/{len(focus_distances)}: {z:.3f} m -> {frames}")
        # render_image appends compositor nodes; clear them so repeated calls in one
        # process do not keep writing to the previous focus folder's file outputs.
        if bpy.context.scene.use_nodes:
            bpy.context.scene.node_tree.nodes.clear()
        render_image(
            camera=camera,
            frames_folder=frames,
            passes_to_save=passes_to_save,
            use_dof=True,
            dof_aperture_fstop=fstop,
            override_num_samples=rgb_samples,
            shifted_camera_num=1,
            merge_duplicate_glass_layers=False,
        )

    if save_gt:
        gt_frames = frames_folder / "gt" / "frames"
        gt_frames.mkdir(parents=True, exist_ok=True)
        camera.data.dof.use_dof = False
        camera.data.shift_x = 0.0
        camera.data.shift_y = 0.0
        if bpy.context.scene.use_nodes:
            bpy.context.scene.node_tree.nodes.clear()
        render_image(
            camera=camera,
            frames_folder=gt_frames,
            passes_to_save=gt_passes_to_save,
            flat_shading=True,
            use_dof=False,
            shifted_camera_num=gt_shifted_camera_num,
            merge_duplicate_glass_layers=merge_duplicate_glass_layers,
            gt_consistency_m=gt_consistency_m,
        )

    metadata = {
        "kind": "dff_focal_stack",
        "scene": scene.name,
        "camera": camera.name,
        "camera_rig_id": camera_rig_id,
        "camera_subcam_id": camera_subcam_id,
        "view_id": (
            applied_view_config.get("view_id")
            if applied_view_config
            else camera.name
        ),
        "source_scene_id": (
            applied_view_config.get("source_scene_id")
            if applied_view_config
            else scene.name
        ),
        "camera_config_path": str(view_config_path) if view_config_path else None,
        "camera_calibration": camera_calibration,
        "blender_version": bpy.app.version_string,
        "render_engine": scene.render.engine,
        "render_device": scene.cycles.device,
        "resolution": [int(res_x), int(res_y)],
        "exposure": round(float(scene.cycles.film_exposure), 3),
        "fill_light_energy": round(float(fill_light_energy), 3),
        "focal_length_mm": round(lens_mm, 3),
        "sensor_width_mm": sensor_width_mm,
        "aperture_fstop": fstop,
        "aperture_blades": int(cam.dof.aperture_blades),
        "focus_spacing": "explicit" if explicit_focus else focus_spacing,
        "depth_percentiles": list(depth_percentiles),
        "depth_window_m": [round(float(depth_window[0]), 4), round(float(depth_window[1]), 4)],
        "depth_source": depth_source,
        "focus_count": len(planes),
        "focus_planes": planes,
        "max_blur_px": round(max_blur, 3),
        "gt_shifted_camera_num": int(gt_shifted_camera_num),
        "gt_consistency_m": float(gt_consistency_m),
        "layer_semantics": (
            "distinct" if merge_duplicate_glass_layers else "bounce"
        ),
        "gt_folder": "gt/frames" if save_gt else None,
        "image_folders": [f"focus_{i:02d}/frames" for i in range(len(planes))],
        "aimed_at": aimed_at,
        "glass_coverage": round(glass_coverage, 4),
        "glass_objects_visible": glass_objects,
    }
    if write_metadata:
        text = json.dumps(metadata, indent=2)
        (frames_folder / f"focal_stack_{camera.name}.json").write_text(text)
        if not (frames_folder / "focal_stack.json").exists():
            # Convenience copy for the common single-camera case.
            (frames_folder / "focal_stack.json").write_text(text)
    return metadata
