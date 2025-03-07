import numpy as np
import random
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go
import plotly.io as pio
from ipywidgets import interact, IntSlider, FloatSlider
from IPython.display import display

# For interactive slider tuning, use the "notebook" renderer.
# (If using Colab, set this to "colab". Note that the "browser" renderer
# does not support interactive ipywidgets.)
pio.renderers.default = 'browser'

def random_direction_in_cone(v, angle_range):
    """
    Generate a random unit vector within a cone of aperture 'angle_range'
    (in radians) around the central direction vector v.
    """
    cos_theta = np.cos(angle_range) + random.random() * (1 - np.cos(angle_range))
    theta = np.arccos(cos_theta)
    phi = random.uniform(0, 2 * np.pi)
    local_dir = np.array([np.sin(theta) * np.cos(phi),
                          np.sin(theta) * np.sin(phi),
                          np.cos(theta)])
    z_axis = np.array([0, 0, 1])
    if np.allclose(v, z_axis):
        return local_dir
    if np.allclose(v, -z_axis):
        rot = R.from_rotvec(np.array([1, 0, 0]) * np.pi)
    else:
        rotation_axis = np.cross(z_axis, v)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        angle = np.arccos(np.dot(z_axis, v))
        rot = R.from_rotvec(rotation_axis * angle)
    new_dir = rot.apply(local_dir)
    return new_dir / np.linalg.norm(new_dir)

def generate_tree_3d(x, y, z, direction, depth, max_depth, branch_length, angle_range_deg, thickness, tree_type, segments, terminals):
    """
    Recursively generate a 3D branching tree.
    """
    if depth >= max_depth:
        terminals.append((x, y, z))
        return

    angle_range = np.deg2rad(angle_range_deg)
    for _ in range(2):
        new_dir = random_direction_in_cone(direction, angle_range)
        x_new = x + branch_length * new_dir[0]
        y_new = y + branch_length * new_dir[1]
        z_new = z + branch_length * new_dir[2]
        segments.append((x, y, z, x_new, y_new, z_new, tree_type, thickness))
        generate_tree_3d(x_new, y_new, z_new, new_dir, depth + 1, max_depth,
                         branch_length * 0.9, angle_range_deg, thickness * 0.8,
                         tree_type, segments, terminals)

def point_to_segment_distance_3d(P, A, B):
    """
    Compute the minimum distance between point P and the line segment AB in 3D.
    """
    P = np.array(P)
    A = np.array(A)
    B = np.array(B)
    AB = B - A
    if np.allclose(AB, 0):
        return np.linalg.norm(P - A)
    t = np.dot(P - A, AB) / np.dot(AB, AB)
    if t < 0:
        return np.linalg.norm(P - A)
    elif t > 1:
        return np.linalg.norm(P - B)
    projection = A + t * AB
    return np.linalg.norm(P - projection)

def nearest_point_on_segment(P, A, B):
    """
    Returns the point on segment AB that is closest to point P.
    """
    P = np.array(P)
    A = np.array(A)
    B = np.array(B)
    AB = B - A
    if np.allclose(AB, 0):
        return A
    t = np.dot(P - A, AB) / np.dot(AB, AB)
    t = max(0, min(1, t))
    return A + t * AB

def generate_curve_points(P0, P1, n_points=20, curvature_factor=0.3):
    """
    Generate points along a quadratic Bézier curve from P0 to P1.
    """
    P0, P1 = np.array(P0), np.array(P1)
    mid = (P0 + P1) / 2
    d = P1 - P0
    length = np.linalg.norm(d)
    if length == 0:
        return np.tile(P0, (n_points, 1))

    random_vec = np.random.randn(3)
    proj = np.dot(random_vec, d) / (length**2) * d
    perp = random_vec - proj
    perp_norm = np.linalg.norm(perp)
    if perp_norm == 0:
        perp = np.zeros(3)
    else:
        perp = perp / perp_norm
    offset = perp * curvature_factor * length
    control = mid + offset

    t_values = np.linspace(0, 1, n_points)
    curve_points = np.outer((1 - t_values) ** 2, P0) \
                   + np.outer(2 * (1 - t_values) * t_values, control) \
                   + np.outer(t_values ** 2, P1)
    return curve_points

def generate_cells(num_cells, possible_cell_types):
    """
    Generate clustered cells for each cell type.
    Returns:
      - cells: array of cell coordinates.
      - cell_types: list of corresponding cell types.
      - cell_requirements: array of nutrient requirements.
    """
    cells = []
    cell_types = []
    cell_requirements = []
    num_types = len(possible_cell_types)
    # Distribute cells roughly equally among types.
    base_num = num_cells // num_types
    remainder = num_cells % num_types

    for i, cell_type in enumerate(possible_cell_types):
        count = base_num + (1 if i < remainder else 0)
        # Use 2 clusters if count is high enough; otherwise 1 cluster.
        clusters = 2 if count > 10 else 1
        cells_per_cluster = count // clusters
        extra = count % clusters
        for c in range(clusters):
            cluster_count = cells_per_cluster + (1 if c < extra else 0)
            # Choose a cluster center within the unit cube (biased toward the center).
            center = np.array([random.uniform(0.2, 0.8) for _ in range(3)])
            std = 0.05  # Standard deviation for the cluster spread.
            for _ in range(cluster_count):
                point = np.random.normal(loc=center, scale=std, size=3)
                point = np.clip(point, 0, 1)
                cells.append(point)
                cell_types.append(cell_type)
                cell_requirements.append(random.uniform(0.6, 0.9))
    cells = np.array(cells)
    cell_requirements = np.array(cell_requirements)
    return cells, cell_types, cell_requirements

def simulate_vasculature_3d(max_depth, angle_range, initial_branch_length,
                            capillary_threshold, num_cells, oxygen_scale,
                            num_arteries, num_veins):
    """
    Generates a 3D vascular network in which:
      - Multiple arterial and venous trees are grown.
      - Capillary sprouts are added iteratively to supply cells.
      - Every arterial and venous terminal is forced to connect via a capillary.
      - Every cell is then connected to both an artery and a vein.
      - Cells (of different types) are generated in clusters to mimic tissue.
    """
    random.seed(42)
    np.random.seed(42)

    segments = []
    arterial_terminals = []
    venous_terminals = []

    # Grow arterial trees along the left boundary.
    for i in range(num_arteries):
        arterial_start = (0,
                          random.uniform(0.3, 0.7),
                          random.uniform(0.3, 0.7))
        arterial_direction = np.array([1, 0, 0])
        generate_tree_3d(arterial_start[0], arterial_start[1], arterial_start[2],
                         arterial_direction, 0, max_depth, initial_branch_length,
                         angle_range, 3.0, 'artery', segments, arterial_terminals)

    # Grow venous trees along the right boundary.
    for i in range(num_veins):
        venous_start = (1,
                        random.uniform(0.3, 0.7),
                        random.uniform(0.3, 0.7))
        venous_direction = np.array([-1, 0, 0])
        generate_tree_3d(venous_start[0], venous_start[1], venous_start[2],
                         venous_direction, 0, max_depth, initial_branch_length,
                         angle_range, 3.0, 'vein', segments, venous_terminals)

    # Initial capillary connections between nearby arterial and venous terminals.
    for A in arterial_terminals:
        for V in venous_terminals:
            dist = np.linalg.norm(np.array(A) - np.array(V))
            if dist < capillary_threshold:
                segments.append((A[0], A[1], A[2],
                                 V[0], V[1], V[2],
                                 'capillary', 0.5))

    # Generate cells (with clustering by type).
    possible_cell_types = ["Type A", "Type B", "Type C"]
    cell_color_map = {"Type A": "purple", "Type B": "orange", "Type C": "teal"}
    cells, cell_types_assigned, cell_requirements = generate_cells(num_cells, possible_cell_types)

    def compute_oxygen_levels(cells, segments):
        oxygen_levels = []
        for cell in cells:
            min_dist = np.inf
            for seg in segments:
                A = seg[0:3]
                B = seg[3:6]
                d = point_to_segment_distance_3d(cell, A, B)
                if d < min_dist:
                    min_dist = d
            oxygen = np.exp(-min_dist / oxygen_scale)
            oxygen_levels.append(oxygen)
        return np.array(oxygen_levels)

    oxygen_levels = compute_oxygen_levels(cells, segments)

    # Iteratively add capillary sprouts for cells that are under-supplied.
    max_iterations = 10
    iteration = 0
    while np.any(oxygen_levels < cell_requirements) and iteration < max_iterations:
        for i, cell in enumerate(cells):
            if oxygen_levels[i] < cell_requirements[i]:
                min_dist = np.inf
                nearest_pt = None
                for seg in segments:
                    A = seg[0:3]
                    B = seg[3:6]
                    pt = nearest_point_on_segment(cell, A, B)
                    d = np.linalg.norm(np.array(cell) - pt)
                    if d < min_dist:
                        min_dist = d
                        nearest_pt = pt
                if nearest_pt is not None:
                    segments.append((nearest_pt[0], nearest_pt[1], nearest_pt[2],
                                     cell[0], cell[1], cell[2],
                                     'capillary', 0.5))
        oxygen_levels = compute_oxygen_levels(cells, segments)
        iteration += 1

    # Ensure every arterial and venous terminal is connected via a capillary.
    def is_terminal_connected(pt, segments, tol=1e-3):
        for seg in segments:
            if seg[6] == 'capillary':
                if np.linalg.norm(np.array(pt) - np.array(seg[0:3])) < tol or \
                   np.linalg.norm(np.array(pt) - np.array(seg[3:6])) < tol:
                    return True
        return False

    for A in arterial_terminals:
        if not is_terminal_connected(A, segments):
            nearest_V = None
            min_dist = float('inf')
            for V in venous_terminals:
                d = np.linalg.norm(np.array(A) - np.array(V))
                if d < min_dist:
                    min_dist = d
                    nearest_V = V
            if nearest_V is not None:
                segments.append((A[0], A[1], A[2],
                                 nearest_V[0], nearest_V[1], nearest_V[2],
                                 'capillary', 0.5))

    for V in venous_terminals:
        if not is_terminal_connected(V, segments):
            nearest_A = None
            min_dist = float('inf')
            for A in arterial_terminals:
                d = np.linalg.norm(np.array(V) - np.array(A))
                if d < min_dist:
                    min_dist = d
                    nearest_A = A
            if nearest_A is not None:
                segments.append((V[0], V[1], V[2],
                                 nearest_A[0], nearest_A[1], nearest_A[2],
                                 'capillary', 0.5))

    # NEW STEP: Ensure every cell is connected to both an artery and a vein.
    for cell in cells:
        # Connect to nearest arterial terminal.
        nearest_A = None
        min_dist_A = float('inf')
        for A in arterial_terminals:
            d = np.linalg.norm(cell - np.array(A))
            if d < min_dist_A:
                min_dist_A = d
                nearest_A = A
        if nearest_A is not None:
            segments.append((nearest_A[0], nearest_A[1], nearest_A[2],
                             cell[0], cell[1], cell[2],
                             'capillary', 0.5))
        # Connect to nearest venous terminal.
        nearest_V = None
        min_dist_V = float('inf')
        for V in venous_terminals:
            d = np.linalg.norm(cell - np.array(V))
            if d < min_dist_V:
                min_dist_V = d
                nearest_V = V
        if nearest_V is not None:
            segments.append((nearest_V[0], nearest_V[1], nearest_V[2],
                             cell[0], cell[1], cell[2],
                             'capillary', 0.5))

    # Prepare smooth vessel curves using quadratic Bézier curves.
    artery_curve_x, artery_curve_y, artery_curve_z = [], [], []
    vein_curve_x, vein_curve_y, vein_curve_z = [], [], []
    capillary_curve_x, capillary_curve_y, capillary_curve_z = [], [], []

    for seg in segments:
        x1, y1, z1, x2, y2, z2, vessel_type, _ = seg
        curvature = 0.1 if vessel_type in ['artery', 'vein'] else 0.3
        curve = generate_curve_points((x1, y1, z1), (x2, y2, z2), n_points=20, curvature_factor=curvature)
        xs, ys, zs = curve[:, 0].tolist(), curve[:, 1].tolist(), curve[:, 2].tolist()
        if vessel_type == 'artery':
            artery_curve_x.extend(xs + [None])
            artery_curve_y.extend(ys + [None])
            artery_curve_z.extend(zs + [None])
        elif vessel_type == 'vein':
            vein_curve_x.extend(xs + [None])
            vein_curve_y.extend(ys + [None])
            vein_curve_z.extend(zs + [None])
        elif vessel_type == 'capillary':
            capillary_curve_x.extend(xs + [None])
            capillary_curve_y.extend(ys + [None])
            capillary_curve_z.extend(zs + [None])

    # Build and display the 3D Plotly figure.
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=artery_curve_x, y=artery_curve_y, z=artery_curve_z,
                               mode='lines',
                               line=dict(color='red', width=4),
                               name='Arteries'))
    fig.add_trace(go.Scatter3d(x=vein_curve_x, y=vein_curve_y, z=vein_curve_z,
                               mode='lines',
                               line=dict(color='blue', width=4),
                               name='Veins'))
    fig.add_trace(go.Scatter3d(x=capillary_curve_x, y=capillary_curve_y, z=capillary_curve_z,
                               mode='lines',
                               line=dict(color='green', width=2),
                               name='Capillaries'))

    # Plot cells grouped by type.
    cells_by_type = {typ: [] for typ in possible_cell_types}
    for i, cell in enumerate(cells):
        cells_by_type[cell_types_assigned[i]].append(cell)

    for typ, pts in cells_by_type.items():
        pts = np.array(pts)
        if pts.shape[0] > 0:
            fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                                       mode='markers',
                                       marker=dict(size=3, color=cell_color_map[typ]),
                                       name=f'Cells {typ}'))

    fig.update_layout(scene=dict(
                        xaxis=dict(title='X', range=[-0.1, 1.1]),
                        yaxis=dict(title='Y', range=[-0.1, 1.1]),
                        zaxis=dict(title='Z', range=[-0.1, 1.1])),
                      title="3D Angiogenesis Simulation with Fully-Connected Vessels & Tissue-like Cell Clustering",
                      width=800, height=800)
    fig.show()

# Call interact (without an __main__ guard) so that sliders work in interactive notebooks.
interact(simulate_vasculature_3d,
         max_depth=IntSlider(min=2, max=8, step=1, value=5, description='Max Depth'),
         angle_range=IntSlider(min=10, max=60, step=1, value=30, description='Angle Range (deg)'),
         initial_branch_length=FloatSlider(min=0.05, max=0.2, step=0.01, value=0.1, description='Initial Branch Length'),
         capillary_threshold=FloatSlider(min=0.05, max=0.5, step=0.01, value=0.15, description='Capillary Threshold'),
         num_cells=IntSlider(min=50, max=500, step=10, value=200, description='Num Cells'),
         oxygen_scale=FloatSlider(min=0.01, max=0.2, step=0.01, value=0.05, description='Oxygen Scale'),
         num_arteries=IntSlider(min=1, max=5, step=1, value=3, description='Num Arteries'),
         num_veins=IntSlider(min=1, max=5, step=1, value=3, description='Num Veins'))
display("Adjust the sliders above to explore parameter effects.")
