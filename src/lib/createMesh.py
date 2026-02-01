import numpy as np
from pathlib import Path
from lib.MeshUtils import load_off, save_off
from lib.config import load_config, parse_args


def meshelize_cuboid(x_range, y_range, z_range, number_vertices):
    w, h, d = x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]
    total_area = (w * h + h * d + w * d) * 2

    # On average, every vertice attarch 6 edges. Each triangle has 3 edges
    mesh_size = total_area / (number_vertices * 2)

    edge_length = (mesh_size * 2) ** 0.5

    x_samples = x_range[0] + np.linspace(0, w, int(w / edge_length + 1))
    y_samples = y_range[0] + np.linspace(0, h, int(h / edge_length + 1))
    z_samples = z_range[0] + np.linspace(0, d, int(d / edge_length + 1))

    xn = x_samples.size
    yn = y_samples.size
    zn = z_samples.size

    out_vertices = []
    out_faces = []
    base_idx = 0

    for n in range(yn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[n], z_samples[0]))
    for m in range(yn - 1):
        for n in range(xn - 1):
            out_faces.append(
                (
                    base_idx + m * xn + n,
                    base_idx + m * xn + n + 1,
                    base_idx + (m + 1) * xn + n,
                ),
            )
            out_faces.append(
                (
                    base_idx + (m + 1) * xn + n + 1,
                    base_idx + m * xn + n + 1,
                    base_idx + (m + 1) * xn + n,
                ),
            )
    base_idx += yn * xn

    for n in range(yn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[n], z_samples[-1]))
    for m in range(yn - 1):
        for n in range(xn - 1):
            out_faces.append(
                (
                    base_idx + m * xn + n,
                    base_idx + m * xn + n + 1,
                    base_idx + (m + 1) * xn + n,
                ),
            )
            out_faces.append(
                (
                    base_idx + (m + 1) * xn + n + 1,
                    base_idx + m * xn + n + 1,
                    base_idx + (m + 1) * xn + n,
                ),
            )
    base_idx += yn * xn

    for n in range(zn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[0], z_samples[n]))
    for m in range(zn - 1):
        for n in range(xn - 1):
            out_faces.append(
                (
                    base_idx + m * xn + n,
                    base_idx + m * xn + n + 1,
                    base_idx + (m + 1) * xn + n,
                ),
            )
            out_faces.append(
                (
                    base_idx + (m + 1) * xn + n + 1,
                    base_idx + m * xn + n + 1,
                    base_idx + (m + 1) * xn + n,
                ),
            )
    base_idx += zn * xn

    for n in range(zn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[-1], z_samples[n]))
    for m in range(zn - 1):
        for n in range(xn - 1):
            out_faces.append(
                (
                    base_idx + m * xn + n,
                    base_idx + m * xn + n + 1,
                    base_idx + (m + 1) * xn + n,
                ),
            )
            out_faces.append(
                (
                    base_idx + (m + 1) * xn + n + 1,
                    base_idx + m * xn + n + 1,
                    base_idx + (m + 1) * xn + n,
                ),
            )
    base_idx += zn * xn

    for n in range(zn):
        for m in range(yn):
            out_vertices.append((x_samples[0], y_samples[m], z_samples[n]))
    for m in range(zn - 1):
        for n in range(yn - 1):
            out_faces.append(
                (
                    base_idx + m * yn + n,
                    base_idx + m * yn + n + 1,
                    base_idx + (m + 1) * yn + n,
                ),
            )
            out_faces.append(
                (
                    base_idx + (m + 1) * yn + n + 1,
                    base_idx + m * yn + n + 1,
                    base_idx + (m + 1) * yn + n,
                ),
            )
    base_idx += zn * yn

    for n in range(zn):
        for m in range(yn):
            out_vertices.append((x_samples[-1], y_samples[m], z_samples[n]))
    for m in range(zn - 1):
        for n in range(yn - 1):
            out_faces.append(
                (
                    base_idx + m * yn + n,
                    base_idx + m * yn + n + 1,
                    base_idx + (m + 1) * yn + n,
                ),
            )
            out_faces.append(
                (
                    base_idx + (m + 1) * yn + n + 1,
                    base_idx + m * yn + n + 1,
                    base_idx + (m + 1) * yn + n,
                ),
            )
    base_idx += zn * yn

    return np.array(out_vertices), np.array(out_faces)

def meshelize_sphere(x_range, y_range, z_range, number_vertices):
    center = np.array([
        (x_range[0] + x_range[1]) / 2,
        (y_range[0] + y_range[1]) / 2,
        (z_range[0] + z_range[1]) / 2
    ])
    a = (x_range[1] - x_range[0]) / 2
    b = (y_range[1] - y_range[0]) / 2
    c = (z_range[1] - z_range[0]) / 2

    radius = max(a, b, c)

    # Create sampling grid
    avg_edge_length = np.sqrt(4 * np.pi * radius**2 / number_vertices)
    n_theta = max(3, int(np.pi * radius / avg_edge_length))
    if n_theta % 2 == 0:
        n_theta += 1
    n_phi = max(3, int(2 * np.pi * radius / avg_edge_length))

    theta = np.linspace(0, np.pi, n_theta, endpoint=True)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    Phi, Theta = np.meshgrid(phi, theta)

    X = radius * np.sin(Theta) * np.cos(Phi) + center[0]
    Y = radius * np.sin(Theta) * np.sin(Phi) + center[1]
    Z = radius * np.cos(Theta) + center[2]

    vertices = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

    # Faces generation similar to ellipsoid
    faces = []
    for i in range(n_theta - 1):
        for j in range(n_phi):
            next_j = (j + 1) % n_phi
            idx0 = i * n_phi + j
            idx1 = (i + 1) * n_phi + j
            idx2 = (i + 1) * n_phi + next_j
            idx3 = i * n_phi + next_j

            faces.append((idx0, idx1, idx2))
            faces.append((idx0, idx2, idx3))

    faces = np.array(faces)

    return vertices, faces


def meshelize_ellipsoid(x_range, y_range, z_range, number_vertices):
    """
    Create a triangulated ellipsoid surface mesh inside the given x, y, z ranges.
    Automatically rescales to match exactly the bounding box.

    Args:
        x_range: (min_x, max_x)
        y_range: (min_y, max_y)
        z_range: (min_z, max_z)
        number_vertices: approximate number of vertices desired

    Returns:
        vertices: (N, 3) numpy array
        faces: (M, 3) numpy array
    """

    # Compute center and initial radii
    center = np.array([
        (x_range[0] + x_range[1]) / 2,
        (y_range[0] + y_range[1]) / 2,
        (z_range[0] + z_range[1]) / 2
    ])
    a = (x_range[1] - x_range[0]) / 2
    b = (y_range[1] - y_range[0]) / 2
    c = (z_range[1] - z_range[0]) / 2

    # Estimate average radius
    avg_radius = (a * b * c) ** (1/3)

    # Estimate approximate triangle size
    sphere_area = 4 * np.pi * avg_radius ** 2
    avg_triangle_area = sphere_area / number_vertices
    avg_edge_length = (2 * avg_triangle_area / np.sqrt(3)) ** 0.5

    # Number of samples
    n_theta = max(3, int(np.pi * avg_radius / avg_edge_length))
    if n_theta % 2 == 0:
        n_theta += 1  # Make odd to sample equator exactly
    n_phi = max(3, int(2 * np.pi * avg_radius / avg_edge_length))

    # Generate theta and phi grids
    theta = np.linspace(0, np.pi, n_theta, endpoint=True)    # polar angle
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)      # azimuthal angle

    Phi, Theta = np.meshgrid(phi, theta)

    # Parametric equation of ellipsoid
    X = a * np.sin(Theta) * np.cos(Phi)
    Y = b * np.sin(Theta) * np.sin(Phi)
    Z = c * np.cos(Theta)

    vertices = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

    # Build faces
    faces = []
    for i in range(n_theta - 1):
        for j in range(n_phi):
            next_j = (j + 1) % n_phi

            idx0 = i * n_phi + j
            idx1 = (i + 1) * n_phi + j
            idx2 = (i + 1) * n_phi + next_j
            idx3 = i * n_phi + next_j

            faces.append((idx0, idx1, idx2))
            faces.append((idx0, idx2, idx3))

    faces = np.array(faces)

    return vertices, faces


def create_mesh(input_path:Path, output_path:Path, number_vertices:int = 1000, linear_coverage:float = 0.99, shape="ellipsoid"):
    # list dir
    for cate_path in input_path.iterdir():
        if not cate_path.is_dir():
            print("Not dir:", cate_path)
            continue
        cate_output_path = output_path / cate_path.name
        cate_output_path.mkdir(exist_ok=True, parents=True)
        f_names = cate_path.glob("*.off")
        f_names = [t.name for t in f_names if len(t.name) < 7]
        vertices = []
        for f_name in f_names:
            vertices_, _ = load_off(cate_path / f_name)
            vertices.append(vertices_)

        vertices = np.concatenate(vertices, axis=0)
        selected_shape = int(vertices.shape[0] * linear_coverage)
        out_pos = []
        for i in range(vertices.shape[1]):
            v_sorted = np.sort(vertices[:, i])
            v_group = v_sorted[selected_shape::] - v_sorted[0:-selected_shape]
            min_idx = np.argmin(v_group)
            out_pos.append((v_sorted[min_idx], v_sorted[min_idx + selected_shape]))
        
        if shape == "sphere":
            xvert, xface = meshelize_sphere(*out_pos, number_vertices=number_vertices)
        elif shape == "ellipsoid":
            xvert, xface = meshelize_ellipsoid(*out_pos, number_vertices=number_vertices)
        elif shape == "cuboid":
            xvert, xface = meshelize_cuboid(*out_pos, number_vertices=number_vertices)
        else:
            raise NotImplementedError("Not implemented")

        save_off(cate_output_path / "01.off", xvert, xface)


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args, load_default_config=False, log_info=False)

    # ellipsoid
    create_mesh(
        input_path=Path(config.dataset.paths.root, "PASCAL3D+_release1.1/CAD"), 
        output_path=Path("/BS/bcos-craft/work/xnovum/src/cuboid_lens/ablation/ellipsoid"),
        number_vertices=1000,
        linear_coverage=0.99
    )


