# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joined terrain with the original Ant environment's grid of starting positions."""

from __future__ import annotations

import numpy as np
import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh


def collision_chunks(mesh: trimesh.Trimesh, max_faces: int = 1_000_000, cell_size: float = 120.0):
    """Group spatial neighbors before splitting, keeping collider bounds local.

    Slicing the face array alone can join the end of one terrain row to the start
    of the next, giving a small collider a map-wide bounding box.

    Avoid excessive fragmentation: global collider/environment pairs scale
    with both collider count and num_envs. A 120m cell at 0.2m resolution has
    720,000 triangles, keeping the 720m terrain to 36 colliders.
    """
    if max_faces <= 0 or cell_size <= 0:
        raise ValueError("max_faces and cell_size must be positive")
    if len(mesh.faces) == 0:
        return
    origin = mesh.bounds[0, :2]
    columns = max(1, int(np.ceil(mesh.extents[1] / cell_size)))
    # Work in batches to avoid materializing all triangle coordinates of a
    # large terrain at once. The retained arrays are just face IDs and bins.
    cells = np.empty(len(mesh.faces), dtype=np.int64)
    for start in range(0, len(mesh.faces), max_faces):
        stop = min(start + max_faces, len(mesh.faces))
        centers = mesh.vertices[mesh.faces[start:stop], :2].mean(axis=1)
        xy = np.floor((centers - origin) / cell_size).astype(np.int64)
        cells[start:stop] = xy[:, 0] * columns + np.minimum(xy[:, 1], columns - 1)
    order = np.argsort(cells, kind="stable")
    sorted_cells = cells[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_cells)) + 1, len(order)]
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        for start in range(begin, end, max_faces):
            faces = mesh.faces[order[start : min(start + max_faces, end)]]
            vertex_ids, indices = np.unique(faces.reshape(-1), return_inverse=True)
            yield trimesh.Trimesh(vertices=mesh.vertices[vertex_ids], faces=indices.reshape(-1, 3), process=False)


def define_mesh(stage, prim_path: str, mesh: trimesh.Trimesh):
    """Author geometry only; the large query mesh must never receive CollisionAPI."""
    from pxr import UsdGeom, Vt

    prim = UsdGeom.Mesh.Define(stage, prim_path)
    prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype=np.float32)))
    prim.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(mesh.faces, dtype=np.int32).reshape(-1)))
    prim.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, dtype=np.int32)))
    prim.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.bounds, dtype=np.float32)))
    prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    return prim


class AntTerrainImporter(TerrainImporter):
    """Use grid origins on generated terrain instead of sampling tile centers."""

    def import_mesh(self, name: str, mesh: trimesh.Trimesh):
        # Height-field tiles share their outermost vertices at z=0. Weld those
        # vertices into one continuous surface; do not leave separate tile edges.
        mesh.merge_vertices()

        origins = self._compute_env_origins_grid(self.cfg.num_envs, self.cfg.env_spacing)
        # Account for terrain underneath the whole starting footprint, rather
        # than placing every robot at world z=0.5 or at an unrelated tile height.
        axis = torch.linspace(-0.5, 0.5, 5, device=self.device)
        x, y = torch.meshgrid(axis, axis, indexing="ij")
        offsets = torch.stack((x.flatten(), y.flatten(), torch.zeros_like(x.flatten())), dim=-1)
        starts = origins[:, None, :] + offsets[None, :, :]
        starts[:, :, 2] = float(mesh.bounds[1, 2]) + 10.0
        directions = torch.zeros_like(starts)
        directions[:, :, 2] = -1.0
        query_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
        hits = raycast_mesh(starts, directions, query_mesh)[0]
        if not torch.isfinite(hits).all():
            raise ValueError("Ant terrain must cover the full env_spacing grid and each robot's starting footprint.")
        origins[:, 2] = hits[:, :, 2].max(dim=1).values
        self._grid_origins = origins
        # The ray caster builds its own mesh later. Release this temporary query
        # mesh before importing the large terrain into the simulation.
        del query_mesh
        self._import_chunked_mesh(name, mesh)

    def _import_chunked_mesh(self, name: str, mesh: trimesh.Trimesh):
        from pxr import Gf, UsdGeom, UsdPhysics

        stage = sim_utils.get_current_stage()
        prim_path = f"{self.cfg.prim_path}/{name}"
        if prim_path in self.terrain_prim_paths:
            raise ValueError(f"Terrain already exists: {prim_path}")
        UsdGeom.Xform.Define(stage, prim_path)
        # Keep the full surface for rendering and Warp ray queries, but never
        # cook it as one collider.
        define_mesh(stage, f"{prim_path}/mesh", mesh)
        if self.cfg.visual_material is not None:
            material_path = f"{prim_path}/visualMaterial"
            self.cfg.visual_material.func(material_path, self.cfg.visual_material)
            sim_utils.bind_visual_material(f"{prim_path}/mesh", material_path)
        physics_material_path = f"{prim_path}/physicsMaterial"
        self.cfg.physics_material.func(physics_material_path, self.cfg.physics_material)
        UsdGeom.Scope.Define(stage, f"{prim_path}/colliders")
        chunk_count = 0
        for chunk_count, chunk in enumerate(collision_chunks(mesh), start=1):
            path = f"{prim_path}/colliders/chunk_{chunk_count:04d}"
            # Put each collider in a local frame instead of authoring every
            # chunk with distant coordinates at the same world-space origin.
            center = chunk.bounds.mean(axis=0)
            chunk.apply_translation(-center)
            collider = define_mesh(stage, path, chunk)
            UsdGeom.Xformable(collider).AddTranslateOp().Set(Gf.Vec3d(*center))
            collider.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            sim_utils.define_collision_properties(path, sim_utils.CollisionPropertiesCfg(collision_enabled=True))
            # Exact static triangles; convex approximation would fill valleys.
            UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim()).CreateApproximationAttr("none")
            sim_utils.bind_physics_material(path, physics_material_path)
        self.terrain_prim_paths.append(prim_path)
        print(f"[INFO] Ant terrain: {len(mesh.faces):,} triangles split into {chunk_count} collision meshes.")

    def configure_env_origins(self, origins=None):
        if self.cfg.terrain_type == "generator":
            self.terrain_origins = None
            self.env_origins = self._grid_origins
        else:
            super().configure_env_origins(origins)
