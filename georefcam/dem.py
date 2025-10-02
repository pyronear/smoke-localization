# === Correction complète pour Ray Tracing DEM avec trimesh ===

import numpy as np
import trimesh
import pyvista as pv
import rasterio
from pyproj import CRS


class RasterioDEM:
    def __init__(self, filepath, crs=None):
        self.filepath = filepath
        self.dataset = rasterio.open(filepath)
        self.crs = crs or CRS.from_string(self.dataset.crs.to_string())
        self.data = self.dataset.read(1)
        self.transform = self.dataset.transform
        self.pcd = None
        self.mesh = None

    def build_pcd(self, sample_step=1):
        rows, cols = self.data.shape
        x_idx = np.arange(0, cols, sample_step)
        y_idx = np.arange(0, rows, sample_step)

        xx, yy = np.meshgrid(x_idx, y_idx)
        xs, ys = rasterio.transform.xy(self.transform, yy, xx)
        xs = np.array(xs).reshape(xx.shape)
        ys = np.array(ys).reshape(yy.shape)
        zs = self.data[yy, xx]

        # Remplacer les NaN/infs dans les altitudes
        zs[~np.isfinite(zs)] = -9999

        self.pcd = np.stack([xs, ys, zs], axis=-1)  # (H, W, 3)

    def build_mesh(self, method="trimesh"):
        if self.pcd is None:
            raise ValueError("Call build_pcd() first")

        if method == "pyvista":
            grid = pv.StructuredGrid(
                self.pcd[:, :, 0], self.pcd[:, :, 1], self.pcd[:, :, 2]
            )
            self.mesh = grid.cast_to_poly_points().delaunay_2d()

        elif method == "trimesh":
            h, w, _ = self.pcd.shape
            vertices = self.pcd.reshape(-1, 3)

            # Faces
            faces = []
            for i in range(h - 1):
                for j in range(w - 1):
                    a = i * w + j
                    b = a + 1
                    c = a + w
                    d = c + 1
                    faces.append([a, b, c])
                    faces.append([b, d, c])
            faces = np.array(faces)

            # Nettoyage manuel
            valid = np.all(np.isfinite(vertices), axis=1)
            index_map = -np.ones(len(valid), dtype=int)
            index_map[valid] = np.arange(np.sum(valid))

            vertices = vertices[valid]
            faces = faces[np.all(valid[faces], axis=1)]
            faces = index_map[faces]

            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

            # Supprimer triangles patho
            mesh.remove_degenerate_faces()
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()

            self.mesh = mesh
            print("mesh.vertices.shape =", self.mesh.vertices.shape)
            print("mesh.vertices[:5] =", self.mesh.vertices[:5])

        else:
            raise ValueError(f"Unknown mesh method: {method}")

    def cast_rays_seq(self, rays):
        if not isinstance(self.mesh, trimesh.Trimesh):
            raise TypeError("Ray tracing requires a trimesh.Trimesh mesh")

        # Flatten the rays to (N, 3)
        ray_origins = rays[:, :, 0].reshape(-1, 3)
        ray_targets = rays[:, :, 1].reshape(-1, 3)
        ray_dirs = ray_targets - ray_origins

        if not np.all(np.isfinite(ray_dirs)):
            invalid = ~np.isfinite(ray_dirs).all(axis=1)
            print("[DEBUG] Non-finite ray_dirs indices:", np.where(invalid)[0])
            print("[DEBUG] Corresponding ray_origins:", ray_origins[invalid])
            print("[DEBUG] Corresponding ray_targets:", ray_targets[invalid])
            raise ValueError("Non-finite values in ray directions before normalization")

        norms = np.linalg.norm(ray_dirs, axis=1, keepdims=True)
        if np.any(norms < 1e-6):
            raise ValueError("Zero-length ray detected in directions")

        ray_dirs = ray_dirs / norms

        if not np.all(np.isfinite(ray_dirs)):
            raise ValueError("Non-finite values in ray directions after normalization")

        engine = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)
        locations, index_ray, _ = engine.intersects_location(
            ray_origins, ray_dirs, multiple_hits=False
        )

        # Fill with NaN first, then set intersection points where available
        inter_points = np.full((ray_origins.shape[0], 3), np.nan)
        inter_points[index_ray] = locations

        return np.stack((ray_origins, inter_points), axis=1)
