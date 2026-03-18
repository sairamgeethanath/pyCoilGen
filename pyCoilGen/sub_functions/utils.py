import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.tri import Triangulation
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
import magpylib as magpy
import trimesh
from shapely.geometry import LineString, Polygon, Point, MultiPolygon
from shapely.ops import unary_union
import os
import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt





def compute_cotangent_laplacian(mesh):
    """
    Compute the cotangent Laplacian matrix for a triangular mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input triangular mesh.

    Returns
    -------
    L : scipy.sparse.csr_matrix
        Sparse cotangent Laplacian matrix (n_vertices x n_vertices).
    """
    verts = mesh.v
    faces = mesh.f
    n = len(verts)

    I = []
    J = []
    V = []

    # Loop over faces
    for tri in faces:
        for i in range(3):
            i0 = tri[i]
            i1 = tri[(i+1)%3]
            i2 = tri[(i+2)%3]

            v0 = verts[i0]
            v1 = verts[i1]
            v2 = verts[i2]

            # Compute cotangent of angle at v0
            u = v1 - v0
            v = v2 - v0
            cot_angle = np.dot(u, v) / np.linalg.norm(np.cross(u, v))

            # Add entries for Laplacian
            I.extend([i1, i2])
            J.extend([i2, i1])
            V.extend([cot_angle, cot_angle])

    # Assemble sparse matrix
    L = csr_matrix((V, (I, J)), shape=(n, n))

    # Make L symmetric
    L = 0.5 * (L + L.T)

    # Compute diagonal
    Ld = np.array(L.sum(axis=1)).flatten()

    # Construct Laplacian: L = D - W
    L = csr_matrix(np.diag(Ld)) - L

    return L


# -----------------------------
# Step 4: Visualization
# -----------------------------
def plot_coil_stream_function(mesh, sf, contour_levels=25, cmap='coolwarm'):
    verts = mesh.v
    faces = mesh.f

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    tri = Triangulation(verts[:,0], verts[:,1], faces)

    # Surface
    surf = ax.plot_trisurf(verts[:,0], verts[:,1], verts[:,2],
                            triangles=faces,
                            cmap=cmap,
                            linewidth=0,
                            antialiased=True,
                            alpha=0.8)
    surf.set_array(sf)
    surf.autoscale()

    # Contours
    ax.tricontour(tri, sf, levels=contour_levels, colors='k', linewidths=1)

    ax.set_axis_off()
    ax.set_title("Stream Function with Coil Contours")
    plt.colorbar(surf, ax=ax, shrink=0.5, label='Stream Function')
    plt.show()

# --------------------------------------------
# final manufacturability spacing verification
# --------------------------------------------


def verify_wire_spacing(wire_loops, wire_spacing):

    if len(wire_loops) < 2:
        return True

    # dense sampling for safety
    dense_paths = []
    for loop in wire_loops:
        p = loop["path"]
        dense_paths.append(p)

    min_distance = np.inf

    for i in range(len(dense_paths)):
        for j in range(i+1, len(dense_paths)):

            tree = cKDTree(dense_paths[j])
            d, _ = tree.query(dense_paths[i], k=1)

            dmin = np.min(d)

            if dmin < min_distance:
                min_distance = dmin

    print(f"\nMinimum wire spacing detected: {min_distance:.6f} m")

    if min_distance < wire_spacing:

        print("WARNING: Wire spacing constraint violated!")
        print(f"Required spacing : {wire_spacing:.6f} m")
        print(f"Detected spacing : {min_distance:.6f} m")

        raise ValueError(
            "Wire spacing constraint violated — design not manufacturable."
        )

    else:

        print("Wire spacing check PASSED.")

    return True





############################################################
# 6 EXTRACT WIRE LOOPS - psi - contours - simple connect - last point first loop - first point next loop
############################################################
def extract_wire_paths_baseline(
        verts,
        faces,
        psi,
        levels,
        display_debug_plots=False,
        smooth_resample=4000):



    # ----------------------------
    # triangulation & contours
    # ----------------------------
    triang = mtri.Triangulation(verts[:,0], verts[:,1], faces)
    fig, ax = plt.subplots()
    cs = ax.tricontour(triang, psi, levels=levels)
    plt.close(fig)

    z_plate = np.mean(verts[:,2])

    # ----------------------------
    # gradient magnitude at vertices (unstructured)
    # ----------------------------
    tree = cKDTree(verts[:,0:2])
    grad_mag = np.zeros(len(verts))
    for i, v in enumerate(verts[:,0:2]):
        dists, idxs = tree.query(v, k=6)
        diffs = psi[idxs] - psi[i]
        grad_mag[i] = np.sqrt(np.sum(diffs**2))

    # ----------------------------
    # helpers
    # ----------------------------
    def signed_area(poly):
        x = poly[:,0]
        y = poly[:,1]
        return 0.5*np.sum(x[:-1]*y[1:] - x[1:]*y[:-1])

    def enforce_orientation(seg, level):
        if np.linalg.norm(seg[0]-seg[-1]) > 1e-9:
            seg = np.vstack([seg, seg[0]])
        area = signed_area(seg)
        # enforce physical current direction
        if level > 0 and area < 0:
            seg = seg[::-1]
        if level < 0 and area > 0:
            seg = seg[::-1]
        return seg

    def smooth_path(path):
        if len(path) > 10:
            try:
                k = min(3, len(path)-1)
                tck,u = splprep([path[:,0], path[:,1], path[:,2]], s=1e-6, k=k)
                u_new = np.linspace(0,1,smooth_resample)
                x,y,z = splev(u_new,tck)
                path = np.column_stack([x,y,z])
            except Exception as e:
                print("Spline smoothing failed:", e)
        return path

    # ----------------------------
    # collect loops
    # ----------------------------
    loops = []
    for level, segs in zip(cs.levels, cs.allsegs):
        for seg in segs:
            if len(seg) < 20:
                continue
            seg = enforce_orientation(seg, level)
            r = np.mean(np.sqrt(seg[:,0]**2 + seg[:,1]**2))
            loops.append({
                "seg": seg,
                "level": level,
                "sign": 1 if level>0 else -1,
                "r": r
            })

    # separate polarities
    pos_loops = sorted([l for l in loops if l["sign"]>0], key=lambda x: x["r"])
    neg_loops = sorted([l for l in loops if l["sign"]<0], key=lambda x: x["r"])

    wire_loops = []

    # ----------------------------
    # connect loops of same polarity by last-to-first point
    # ----------------------------
    def build_path(loopset):
        if len(loopset)==0:
            return None
        full_path = []
        prev_exit = None
        for loop in loopset:
            seg = loop["seg"]
            path3d = np.column_stack([
                seg[:,0],
                seg[:,1],
                np.full(len(seg), z_plate)
            ])

            # simple bridge from last to first
            if prev_exit is not None:
                start = prev_exit
                attach = path3d[0]
                bridge = np.vstack([start, attach])
                full_path.append(bridge)

            full_path.append(path3d)
            prev_exit = path3d[-1]

        path = np.vstack(full_path)
        path = smooth_path(path)
        return path

    pos_path = build_path(pos_loops)
    neg_path = build_path(neg_loops)

    if pos_path is not None:
        wire_loops.append({"path":pos_path,"sign":1})
    if neg_path is not None:
        wire_loops.append({"path":neg_path,"sign":-1})

    # ----------------------------
    # debug plot
    # ----------------------------
    if display_debug_plots:
        fig, ax = plt.subplots(figsize=(6,6))
        for loop in wire_loops:
            p = loop["path"]
            ax.plot(p[:,0], p[:,1], 'r' if loop["sign"]>0 else 'b')
        ax.set_aspect("equal")
        ax.set_title("Gradient Coil Wire Paths")
        plt.show()

    return wire_loops

# ==========================================================
# Extract wire path with wire spacing 
# ==========================================================
def extract_wire_paths(
        verts,
        faces,
        psi,
        levels,
        display_debug_plots=False,
        smooth_resample=4000,
        wire_width=0.002,
        wire_thickness=0.0016,
        clearance=0.0005,
        terminal_cut_length=0.0015):

    import numpy as np
    import matplotlib.tri as mtri
    import matplotlib.pyplot as plt
    from scipy.interpolate import splprep, splev
    from scipy.spatial import cKDTree
    import pyvista as pv

    z_plate = np.mean(verts[:, 2])

    # Minimum spacing between wire centerlines
    min_spacing = wire_width + clearance + 0.25 * wire_width

    triang = mtri.Triangulation(verts[:, 0], verts[:, 1], faces)

    fig, ax = plt.subplots()
    cs = ax.tricontour(triang, psi, levels=levels)
    plt.close(fig)

    # ------------------------------------------------
    # Smooth path
    # ------------------------------------------------
    def smooth_path(path):

        if len(path) < 10:
            return path

        try:

            k = min(3, len(path) - 1)

            tck, u = splprep(
                [path[:,0], path[:,1], path[:,2]],
                s=1e-6,
                k=k
            )

            u_new = np.linspace(0,1,smooth_resample)

            x,y,z = splev(u_new,tck)

            return np.column_stack([x,y,z])

        except:
            return path

    # ------------------------------------------------
    # Arc bridge
    # ------------------------------------------------
    def arc_bridge(p1, p2, z, npts=40):

        r1 = np.sqrt(p1[0]**2 + p1[1]**2)
        r2 = np.sqrt(p2[0]**2 + p2[1]**2)

        r = 0.5 * (r1 + r2)

        th1 = np.arctan2(p1[1], p1[0])
        th2 = np.arctan2(p2[1], p2[0])

        dth = th2 - th1

        if dth > np.pi:
            dth -= 2*np.pi
        if dth < -np.pi:
            dth += 2*np.pi

        th = np.linspace(th1, th1 + dth, npts)

        x = r * np.cos(th)
        y = r * np.sin(th)

        return np.column_stack([x, y, np.full(npts, z)])

    # ------------------------------------------------
    # Cut terminal end (NEW)
    # ------------------------------------------------
    def cut_terminal_end(path, cut_length):

        if len(path) < 5:
            return path

        seg_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)

        total = 0.0
        idx = len(path) - 1

        while idx > 1 and total < cut_length:
            total += seg_lengths[idx-1]
            idx -= 1

        return path[:idx]

    # ------------------------------------------------
    # Extract loops
    # ------------------------------------------------
    loops = []

    for level, segs in zip(cs.levels, cs.allsegs):

        for seg in segs:

            if len(seg) < 30:
                continue

            r = np.mean(np.sqrt(seg[:,0]**2 + seg[:,1]**2))

            path = np.column_stack([
                seg[:,0],
                seg[:,1],
                np.full(len(seg), z_plate)
            ])

            loops.append({
                "path": path,
                "sign": 1 if level > 0 else -1,
                "r": r
            })

    # Separate polarities
    pos_loops = sorted([l for l in loops if l["sign"] > 0], key=lambda x: x["r"])
    neg_loops = sorted([l for l in loops if l["sign"] < 0], key=lambda x: x["r"])

    # ------------------------------------------------
    # Enforce global spacing
    # ------------------------------------------------
    def enforce_spacing(loopset):

        accepted = []
        global_pts = None

        for loop in loopset:

            pts = loop["path"]

            if global_pts is None:

                accepted.append(pts)
                global_pts = pts

                continue

            tree = cKDTree(global_pts)

            d,_ = tree.query(pts)

            if np.min(d) > min_spacing:

                accepted.append(pts)
                global_pts = np.vstack([global_pts, pts])

        return accepted

    pos_loops = enforce_spacing(pos_loops)
    neg_loops = enforce_spacing(neg_loops)

    # ------------------------------------------------
    # Build spiral
    # ------------------------------------------------
    def build_spiral(loopset):

        if len(loopset) == 0:
            return None

        spiral = loopset[0].copy()

        cut_pts = max(5, int(0.16 * len(spiral)))
        spiral = spiral[:-cut_pts]

        for loop in loopset[1:]:

            prev_end = spiral[-1]

            tree = cKDTree(loop)

            _, idx = tree.query(prev_end)

            loop = np.roll(loop, -idx, axis=0)

            bridge = arc_bridge(prev_end, loop[0], z_plate)

            spiral = np.vstack([spiral, bridge, loop])

        spiral = smooth_path(spiral)

        # NEW: trim terminal end
        spiral = cut_terminal_end(spiral, terminal_cut_length)

        return spiral

    wire_loops = []

    pos_spiral = build_spiral(pos_loops)
    neg_spiral = build_spiral(neg_loops)

    if pos_spiral is not None:
        wire_loops.append({
            "path": pos_spiral,
            "sign": 1,
            "width": wire_width,
            "thickness": wire_thickness
        })

    if neg_spiral is not None:
        wire_loops.append({
            "path": neg_spiral,
            "sign": -1,
            "width": wire_width,
            "thickness": wire_thickness
        })

    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------
    if display_debug_plots:

        plotter = pv.Plotter()

        for loop in wire_loops:

            poly = pv.lines_from_points(loop["path"])

            tube = poly.tube(radius=wire_width/2)

            plotter.add_mesh(
                tube,
                color="red" if loop["sign"] > 0 else "blue"
            )

        plotter.add_axes()
        plotter.show_grid()
        plotter.show()

    return wire_loops

# ==========================================================
# Extract wire path with wire spacing and simple bridge connections between loops of same polarity
# ==========================================================
def extract_wire_paths_suboptimal_bridges(
        verts,
        faces,
        psi,
        levels,
        display_debug_plots=False,
        smooth_resample=4000,
        wire_width=0.002,
        wire_thickness=0.0016,
        clearance=0.0005):

    import numpy as np
    import matplotlib.tri as mtri
    import matplotlib.pyplot as plt
    from scipy.interpolate import splprep, splev
    from scipy.spatial import cKDTree
    import pyvista as pv

    z_plate = np.mean(verts[:, 2])

    # Minimum spacing between wire centerlines
    min_spacing = wire_width + clearance + 0.25 * wire_width

    triang = mtri.Triangulation(verts[:, 0], verts[:, 1], faces)

    fig, ax = plt.subplots()
    cs = ax.tricontour(triang, psi, levels=levels)
    plt.close(fig)

    # ------------------------------------------------
    # Smooth path
    # ------------------------------------------------

    def smooth_path(path):

        if len(path) < 10:
            return path

        try:
            k = min(3, len(path) - 1)

            tck, u = splprep(
                [path[:,0], path[:,1], path[:,2]],
                s=1e-6,
                k=k
            )

            u_new = np.linspace(0,1,smooth_resample)

            x,y,z = splev(u_new,tck)

            return np.column_stack([x,y,z])

        except:
            return path

    # ------------------------------------------------
    # Extract loops
    # ------------------------------------------------

    loops = []

    for level, segs in zip(cs.levels, cs.allsegs):

        for seg in segs:

            if len(seg) < 30:
                continue

            r = np.mean(np.sqrt(seg[:,0]**2 + seg[:,1]**2))

            path = np.column_stack([
                seg[:,0],
                seg[:,1],
                np.full(len(seg), z_plate)
            ])

            loops.append({
                "path": path,
                "sign": 1 if level > 0 else -1,
                "r": r
            })

    # Separate polarities
    pos_loops = sorted([l for l in loops if l["sign"] > 0], key=lambda x: x["r"])
    neg_loops = sorted([l for l in loops if l["sign"] < 0], key=lambda x: x["r"])

    # ------------------------------------------------
    # Enforce global spacing
    # ------------------------------------------------

    def enforce_spacing(loopset):

        accepted = []
        global_pts = None

        for loop in loopset:

            pts = loop["path"]

            if global_pts is None:

                accepted.append(pts)
                global_pts = pts

                continue

            tree = cKDTree(global_pts)

            d,_ = tree.query(pts)

            if np.min(d) > min_spacing:

                accepted.append(pts)
                global_pts = np.vstack([global_pts, pts])

        return accepted

    pos_loops = enforce_spacing(pos_loops)
    neg_loops = enforce_spacing(neg_loops)

    # ------------------------------------------------
    # Build spiral (with center loop opening)
    # ------------------------------------------------

    def build_spiral(loopset):

        if len(loopset) == 0:
            return None

        spiral = loopset[0].copy()

        # OPEN inner loop to avoid center overlap
        cut_pts = max(5, int(0.16 * len(spiral))) # cut 20% of points
        spiral = spiral[:-cut_pts]

        for loop in loopset[1:]:

            prev_end = spiral[-1]

            tree = cKDTree(loop)

            _, idx = tree.query(prev_end)

            loop = np.roll(loop, -idx, axis=0)

            bridge = np.linspace(prev_end, loop[0], 10)

            spiral = np.vstack([spiral, bridge, loop])

        spiral = smooth_path(spiral)

        return spiral

    wire_loops = []

    pos_spiral = build_spiral(pos_loops)
    neg_spiral = build_spiral(neg_loops)

    if pos_spiral is not None:
        wire_loops.append({
            "path": pos_spiral,
            "sign": 1,
            "width": wire_width,
            "thickness": wire_thickness
        })

    if neg_spiral is not None:
        wire_loops.append({
            "path": neg_spiral,
            "sign": -1,
            "width": wire_width,
            "thickness": wire_thickness
        })

    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------

    if display_debug_plots:

        plotter = pv.Plotter()

        for loop in wire_loops:

            poly = pv.lines_from_points(loop["path"])

            tube = poly.tube(radius=wire_width/2)

            plotter.add_mesh(
                tube,
                color="red" if loop["sign"] > 0 else "blue"
            )

        plotter.add_axes()
        plotter.show()

    return wire_loops
# ==========================================================
# Simulate using magpylib
# ==========================================================
def simulate_gradient_coil(coil_parts, DSV_IMAGING, wire, display_field=False):
    """
    Simulate gradient coil B-field including wire width and thickness.
    
    Parameters
    ----------
    coil_parts : list
        List of coil parts, each having `wire_loops` extracted from extract_wire_paths.
    DSV_IMAGING : float
        Diameter of the imaging sphere in meters.
    wire : dict
        Contains {"current": <float>} in Amperes.
    display_field : bool
        If True, display 3D scatter plots of Bx, By, Bz.
    
    Returns
    -------
    dict
        {"points": points, "B": B, "Bx": Bx, "By": By, "Bz": Bz}
    """

    import numpy as np
    import magpylib as magpy
    import matplotlib.pyplot as plt

    # -----------------------------
    # Create imaging sphere points
    # -----------------------------
    r = DSV_IMAGING / 2
    grid = np.linspace(-r, r, 21)
    points = np.array([[x, y, z]
                       for x in grid for y in grid for z in grid
                       if x**2 + y**2 + z**2 <= r**2])

    current_scale = wire["current"]
    B = np.zeros((len(points), 3))

    # -----------------------------
    # Loop over coil parts and loops
    # -----------------------------
    for part in coil_parts:
        for spiral in part.wire_loops:
            path = spiral["path"]
            width = spiral.get("width", wire['width']) # default to thickness if width not specified
            thickness = spiral.get("thickness", wire['thickness']) # default to width if thickness not specified

            # Downsample if spline too long (memory efficiency)
            if len(path) > 400:
                idx = np.linspace(0, len(path)-1, 400).astype(int)
                path = path[idx]

            # 4-point rectangular quadrature to approximate wire cross-section
            offsets = [(-width/2, -thickness/2),
                       (-width/2,  thickness/2),
                       ( width/2, -thickness/2),
                       ( width/2,  thickness/2)]

            for dx, dz in offsets:
                offset_path = path.copy()
                offset_path[:, 0] += dx
                offset_path[:, 2] += dz

                # Polyline source (current direction taken from vertex order)
                source = magpy.current.Polyline(
                    current=current_scale / len(offsets),  # divided among offsets
                    vertices=offset_path
                )

                # accumulate B-field
                B += magpy.getB(source, points)

    # -----------------------------
    # Extract components
    # -----------------------------
    Bx = B[:, 0]
    By = B[:, 1]
    Bz = B[:, 2]

    # -----------------------------
    # Optional visualization
    # -----------------------------
    if display_field:
        fig = plt.figure(figsize=(15,5))
        for i, (comp, title) in enumerate(zip([Bx, By, Bz], ["Bx","By","Bz"])):
            ax = fig.add_subplot(1, 3, i+1, projection='3d')
            sc = ax.scatter(points[:,0], points[:,1], points[:,2], c=comp, cmap='jet')
            ax.set_title(title)
            fig.colorbar(sc, ax=ax, shrink=0.6)
        fig.suptitle("Simulated Gradient Field")
        plt.tight_layout()
        plt.show()

    return {"points": points, "B": B, "Bx": Bx, "By": By, "Bz": Bz}
# ==========================================================
# Simulate using magpylib - does not consider wire thickness - just centerline current
# ==========================================================
def simulate_gradient_coil_polylines(coil_parts, DSV_IMAGING, wire, display_field=False):



    r = DSV_IMAGING / 2
    grid = np.linspace(-r, r, 21)

    points = np.array([
        [x, y, z]
        for x in grid
        for y in grid
        for z in grid
        if x**2 + y**2 + z**2 <= r**2
    ])

    current_scale = wire["current"]

    B = np.zeros((len(points),3))

    sources = []

    for part in coil_parts:

        for spiral in part.wire_loops:

            path = spiral["path"]
            polarity = spiral["sign"]

            # current = polarity * current_scale

            wire_source = magpy.current.Polyline(
                current=current_scale, # the direction is misleading as the vertices of the loop determine this anyway - avoid double negative
                vertices=path
            )

            sources.append(wire_source)

    # vectorized field evaluation
    B_total = magpy.getB(sources, points)

    B = np.sum(B_total, axis=0)

    Bx = B[:,0]
    By = B[:,1]
    Bz = B[:,2]

    if display_field:

        fig = plt.figure(figsize=(15,5))

        ax1 = fig.add_subplot(131, projection='3d')
        sc1 = ax1.scatter(points[:,0], points[:,1], points[:,2], c=Bx, cmap='jet')
        ax1.set_title("Bx")
        fig.colorbar(sc1, ax=ax1, shrink=0.6)

        ax2 = fig.add_subplot(132, projection='3d')
        sc2 = ax2.scatter(points[:,0], points[:,1], points[:,2], c=By, cmap='jet')
        ax2.set_title("By")
        fig.colorbar(sc2, ax=ax2, shrink=0.6)

        ax3 = fig.add_subplot(133, projection='3d')
        sc3 = ax3.scatter(points[:,0], points[:,1], points[:,2], c=Bz, cmap='jet')
        ax3.set_title("Bz")
        fig.colorbar(sc3, ax=ax3, shrink=0.6)

        fig.suptitle("Simulated Gradient Field")
        plt.tight_layout()
        plt.show()

    return {
        "points": points,
        "B": B,
        "Bx": Bx,
        "By": By,
        "Bz": Bz
    }


# ==========================================================
# 3D printer G-code generation for gradient coil holder
# ==========================================================

def generate_gradient_former(
        coil_parts,
        output_prefix="gradient_plate",
        plate_diameter_m=0.1524,
        plate_thickness_m=0.003,
        wire_width_m=0.002,
        display_debug_plots=True,
        save_loop_coords=True):

    """
    Export gradient coil loops as RS-274X Gerber for CNC fabrication
    and provide 2D visualization including wire width for overlap checks.

    Also exports complement Gerber files where the copper region is the
    inverse of the wire paths for isolation milling.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    mm = 1000.0
    plate_radius_mm = plate_diameter_m*mm/2
    wire_width_mm = wire_width_m*mm

    #-------------------------
    # Generate the plate STL for merging later if needed
    #-------------------------



    # ----------------------------------------
    # Helper to ensure minimum spacing between points
    # ----------------------------------------
    def enforce_global_spacing(points, min_dist_mm=0.1):
        filtered=[points[0]]
        for p in points[1:]:
            if np.linalg.norm(p-filtered[-1])>min_dist_mm:
                filtered.append(p)
        return np.array(filtered)

    # ----------------------------------------
    # Helper to generate circular plate
    # ----------------------------------------
    def generate_plate_circle(radius_mm, npts=720):
        th = np.linspace(0,2*np.pi,npts)
        x = radius_mm*np.cos(th)
        y = radius_mm*np.sin(th)
        return np.column_stack([x,y])

    # ----------------------------------------
    # Process each coil plate
    # ----------------------------------------
    for part_index, part in enumerate(coil_parts):

        print(f"\nProcessing plate {part_index}")

        gerber_paths = []

        # ----------------------------------------
        # Collect paths
        # ----------------------------------------
        for loop_index, loop in enumerate(part.wire_loops):

            pts_mm = np.asarray(loop["path"])[:, :2]*mm
            pts_mm = enforce_global_spacing(pts_mm, min_dist_mm=0.5)

            gerber_paths.append((pts_mm, loop["sign"]))

            if save_loop_coords:
                coords = np.column_stack([
                    pts_mm[:,0],
                    pts_mm[:,1],
                    np.zeros(len(pts_mm))
                ])

                fname = f"{output_prefix}_plate{part_index}_loop{loop_index}.txt"
                np.savetxt(fname, coords, fmt="%.2f")



        # ----------------------------------------
        # Write standard Gerber (tracks)
        # ----------------------------------------
        gerber_fname = f"{output_prefix}_plate{part_index}.gbr"

        with open(gerber_fname, "w") as f:

            f.write("G04 Gradient coil generated by Python*\n")
            f.write("%MOMM*%\n")
            f.write("%FSLAX24Y24*%\n")

            f.write(f"%ADD10C,{wire_width_mm:.4f}*%\n")

            f.write("D10*\n")

            for path, sign in gerber_paths:

                x0, y0 = path[0]

                f.write(f"X{int(x0*10000):07d}Y{int(y0*10000):07d}D02*\n")

                for p in path[1:]:
                    x, y = p
                    f.write(f"X{int(x*10000):07d}Y{int(y*10000):07d}D01*\n")

            f.write("M02*\n")

        print("Saved Gerber:", gerber_fname)

        # ----------------------------------------
        # Write complement Gerber (plate minus tracks)
        # ----------------------------------------
        complement_fname = f"{output_prefix}_plate{part_index}_complement.gbr"

        plate_circle = generate_plate_circle(plate_radius_mm)

        with open(complement_fname, "w") as f:

            f.write("G04 Complement copper layer*\n")
            f.write("%MOMM*%\n")
            f.write("%FSLAX24Y24*%\n")

            # aperture for tracks
            f.write(f"%ADD10C,{wire_width_mm:.4f}*%\n")

            # ---- Filled plate region ----
            f.write("%LPD*%\n")
            f.write("G36*\n")

            x0, y0 = plate_circle[0]

            f.write(f"X{int(x0*10000):07d}Y{int(y0*10000):07d}D02*\n")

            for p in plate_circle[1:]:
                x,y = p
                f.write(f"X{int(x*10000):07d}Y{int(y*10000):07d}D01*\n")

            f.write(f"X{int(x0*10000):07d}Y{int(y0*10000):07d}D01*\n")

            f.write("G37*\n")

            # ---- Subtract wire paths ----
            f.write("%LPC*%\n")
            f.write("D10*\n")

            for path,_ in gerber_paths:

                x0,y0 = path[0]

                f.write(f"X{int(x0*10000):07d}Y{int(y0*10000):07d}D02*\n")

                for p in path[1:]:
                    x,y = p
                    f.write(f"X{int(x*10000):07d}Y{int(y*10000):07d}D01*\n")

            f.write("M02*\n")

        print("Saved Complement Gerber:", complement_fname)

        # ----------------------------------------
        # Visualization
        # ----------------------------------------
        if display_debug_plots:

            plt.figure(figsize=(6,6))
            ax = plt.gca()

            for path, sign in gerber_paths:

                ax.plot(
                    path[:,0],
                    path[:,1],
                    color="red" if sign>0 else "blue",
                    linewidth=1.5
                )

                for i in range(len(path)-1):

                    p0, p1 = path[i], path[i+1]

                    dx, dy = p1[0]-p0[0], p1[1]-p0[1]

                    length = np.sqrt(dx**2 + dy**2)

                    angle = np.degrees(np.arctan2(dy, dx))

                    from matplotlib.patches import Rectangle

                    rect = Rectangle(
                        (p0[0]-wire_width_mm/2, p0[1]-wire_width_mm/2),
                        length,
                        wire_width_mm,
                        angle=angle,
                        color="red" if sign>0 else "blue",
                        alpha=0.2
                    )

                    ax.add_patch(rect)

            circle = plt.Circle(
                (0,0),
                plate_radius_mm,
                fill=False,
                linestyle='--',
                color='gray'
            )

            ax.add_patch(circle)

            ax.set_aspect('equal')
            ax.set_xlabel("X [mm]")
            ax.set_ylabel("Y [mm]")

            plt.title(
                f"Plate {part_index} Wire Loops with {wire_width_mm:.1f} mm Width"
            )

            plt.show()
        # ----------------------------------------
        # STL GENERATION (FINAL: all features)
        # ----------------------------------------
        try:
            import trimesh
            from scipy.interpolate import splprep, splev
            import numpy as np

            def smooth_path(x, y, smoothing=5.0, n_points=300):
                tck, _ = splprep([x, y], s=smoothing, per=True)
                u = np.linspace(0, 1, n_points)
                x_new, y_new = splev(u, tck)
                return np.array(x_new), np.array(y_new)

            all_segments = []
            hole_meshes = []
            strain_relief_segments = []
            numbering_meshes = []

            plate_height_mm = plate_thickness_m * mm
            hole_radius = wire_width_mm * 0.8
            hole_height = plate_height_mm * 2.0
            offset_dist = wire_width_mm * 1.5
            strain_length = wire_width_mm * 4.0
            tube_radius = wire_width_mm / 2.0

            # ---- Plate ----
            plate = trimesh.creation.cylinder(
                radius=plate_radius_mm,
                height=plate_height_mm,
                sections=128
            )
            plate.apply_translation([0, 0, plate_height_mm / 2])

            # ---- Alignment pins (snap-fit) ----
            pin_radius = wire_width_mm
            pin_height = plate_height_mm * 1.2
            pin_offset = plate_radius_mm * 0.85
            pin_angles = [0, np.pi/2, np.pi, 3*np.pi/2]  # 4 pins

            pin_meshes = []
            for a in pin_angles:
                x = pin_offset * np.cos(a)
                y = pin_offset * np.sin(a)
                pin = trimesh.creation.cylinder(
                    radius=pin_radius,
                    height=pin_height
                )
                pin.apply_translation([x, y, plate_height_mm / 2])
                pin_meshes.append(pin)

            # ---- Process each loop ----
            for loop_index, (path, sign) in enumerate(gerber_paths):

                x_raw, y_raw = path[:,0], path[:,1]
                points = smooth_path(x_raw, y_raw, smoothing=5.0)

                points_3d = np.column_stack([points[0], points[1], np.zeros_like(points[0])])

                # ---- Tube segments ----
                for i in range(len(points_3d)-1):
                    p1, p2 = points_3d[i], points_3d[i+1]
                    if np.linalg.norm(p2 - p1) < 0.5:
                        continue
                    seg = trimesh.creation.cylinder(
                        radius=tube_radius,
                        segment=[p1, p2]
                    )
                    # optional multi-layer: raise every other loop slightly
                    if loop_index % 2 == 1:
                        seg.apply_translation([0,0,0.2])
                    all_segments.append(seg)

                # ---- Terminal holes ----
                def tangent(p0, p1):
                    v = p1 - p0
                    n = np.linalg.norm(v)
                    return v/n if n>0 else np.array([1.0,0.0])
                def perp(v): return np.array([-v[1], v[0]])

                t_start = tangent(path[0], path[1])
                t_end = tangent(path[-2], path[-1])
                start_pt = path[0] + perp(t_start)*offset_dist
                end_pt = path[-1] + perp(t_end)*offset_dist

                for pt in [start_pt, end_pt]:
                    hole = trimesh.creation.cylinder(
                        radius=hole_radius,
                        height=hole_height
                    )
                    hole.apply_translation([pt[0], pt[1], plate_height_mm / 2])
                    hole_meshes.append(hole)

                # ---- Strain relief ----
                for base_pt, t_vec in [(start_pt, t_start), (end_pt, t_end)]:
                    p1 = np.array([base_pt[0], base_pt[1], 0])
                    p2 = p1 + np.append(t_vec * strain_length, 0)
                    relief = trimesh.creation.cylinder(
                        radius=tube_radius,
                        segment=[p1, p2]
                    )
                    strain_relief_segments.append(relief)

                # ---- Loop numbering ----
                try:
                    num_text = trimesh.creation.text(
                        text=str(loop_index+1),
                        height=wire_width_mm*2,
                        depth=0.5
                    )
                    # place number at first point of loop, slightly above plate
                    num_text.apply_translation([
                        path[0][0],
                        path[0][1],
                        plate_height_mm - 0.3
                    ])
                    numbering_meshes.append(num_text)
                except Exception as e:
                    print(f"⚠️ Loop numbering failed for loop {loop_index+1}: {e}")

            # ---- Combine all cutters ----
            tube = trimesh.util.concatenate(all_segments)
            if strain_relief_segments:
                relief_mesh = trimesh.util.concatenate(strain_relief_segments)
                tube = trimesh.util.concatenate([tube, relief_mesh])
            cutter_meshes = [tube] + hole_meshes + numbering_meshes + pin_meshes
            cutter = trimesh.util.concatenate(cutter_meshes)

            # ---- Offset tube slightly into plate for engraving/groove ----
            cutter.apply_translation([0,0,plate_height_mm*0.75])

            # ---- Plate engraving ----
            label = "TOP" if part_index==0 else "BOTTOM"
            try:
                text_mesh = trimesh.creation.text(
                    text=label,
                    height=plate_radius_mm*0.15,
                    depth=0.5
                )
                text_mesh.apply_translation([
                    -plate_radius_mm*0.3,
                    -plate_radius_mm*0.8,
                    plate_height_mm - 0.5
                ])
                cutter = trimesh.util.concatenate([cutter, text_mesh])
            except:
                print("⚠️ Plate engraving skipped")

            # ---- Boolean difference ----
            result = plate.difference(cutter)
            if result is None or len(result.vertices)==0:
                print("❌ Boolean failed")
                continue

            # ---- Export STL ----
            stl_fname = f"{output_prefix}_plate{part_index}_final_print.stl"
            result.export(stl_fname)
            print(f"✅ Final STL saved: {stl_fname}")


            # ----------------------------------------
            # DEBUG VISUALIZATION (Matplotlib + terminals + strain relief + loop numbers)
            # ----------------------------------------
            if display_debug_plots:
                import matplotlib.pyplot as plt
                from matplotlib.patches import Rectangle, Circle, FancyArrow

                plt.figure(figsize=(8,8))
                ax = plt.gca()

                # Draw plate boundary
                circle = Circle(
                    (0,0),
                    plate_radius_mm,
                    fill=False,
                    linestyle='--',
                    color='gray',
                    linewidth=2
                )
                ax.add_patch(circle)

                # Draw wire loops
                for loop_index, (path, sign) in enumerate(gerber_paths):
                    ax.plot(
                        path[:,0],
                        path[:,1],
                        color="red" if sign>0 else "blue",
                        linewidth=1.5,
                        label=f"Loop {loop_index+1}"
                    )

                    # Draw grooves as semi-transparent rectangles
                    for i in range(len(path)-1):
                        p0, p1 = path[i], path[i+1]
                        dx, dy = p1-p0
                        length = np.sqrt(dx**2 + dy**2)
                        angle = np.degrees(np.arctan2(dy, dx))
                        rect = Rectangle(
                            (p0[0]-wire_width_mm/2, p0[1]-wire_width_mm/2),
                            length,
                            wire_width_mm,
                            angle=angle,
                            color="red" if sign>0 else "blue",
                            alpha=0.2
                        )
                        ax.add_patch(rect)

                    # Draw start and end terminal holes
                    def tangent(p0, p1):
                        v = p1 - p0
                        n = np.linalg.norm(v)
                        return v/n if n>0 else np.array([1.0,0.0])
                    def perp(v): return np.array([-v[1], v[0]])

                    t_start = tangent(path[0], path[1])
                    t_end = tangent(path[-2], path[-1])
                    start_pt = path[0] + perp(t_start)*wire_width_mm*1.5
                    end_pt   = path[-1] + perp(t_end)*wire_width_mm*1.5

                    ax.add_patch(Circle(start_pt, radius=wire_width_mm*0.8, color='green', alpha=0.5))
                    ax.add_patch(Circle(end_pt, radius=wire_width_mm*0.8, color='green', alpha=0.5))

                    # Draw strain relief arrows
                    strain_len = wire_width_mm*4
                    ax.add_patch(FancyArrow(
                        start_pt[0], start_pt[1],
                        t_start[0]*strain_len, t_start[1]*strain_len,
                        width=0.3, color='orange', alpha=0.6
                    ))
                    ax.add_patch(FancyArrow(
                        end_pt[0], end_pt[1],
                        t_end[0]*strain_len, t_end[1]*strain_len,
                        width=0.3, color='orange', alpha=0.6
                    ))

                    # Draw loop numbers
                    ax.text(path[0][0], path[0][1], str(loop_index+1),
                            fontsize=10, color='purple', weight='bold')

                    ax.set_aspect('equal')
                    ax.set_xlabel("X [mm]")
                    ax.set_ylabel("Y [mm]")
                    plt.title(f"Plate {part_index} Wire Loops with Terminals and Strain Relief")
                    plt.grid(True)
                plt.show()
        except Exception as e:
            print("❌ STL generation failed:", e)

    print("\nGradient former coordinate export complete with Gerber and visualization.")

    # ==========================================================
# PRINT PERFORMANCE METRICS
# ==========================================================

def print_metrics(Bz, coords, axis = 'x'):

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=Bz * 1e3, cmap='jet')
    plt.colorbar(sc, label='Gradient Field (mT)')
    ax.set_title('Optimized: ' + axis + ' Magnetic Gradient Field')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.show()

    # Evaluate along X-axis
    if axis == 'x':
        axis_mask = (np.abs(y) < 1e-3) & (np.abs(z) < 1e-3)
        x_axis = x[axis_mask]
        g_axis = Bz[axis_mask]
        
        coeffs = np.polyfit(x_axis, g_axis, 1)
        slope = coeffs[0]

        print("\n===== X GRADIENT PERFORMANCE =====")
        print(f"Gradient efficiency (mT/m/A): {slope*1e3:.3f}")

        linear_fit = np.polyval(coeffs, x_axis)
        error = (g_axis - linear_fit) / np.max(np.abs(linear_fit)) * 100
        print(f"Max linearity error within DSV (%): {np.max(np.abs(error)):.2f}")

    elif axis == 'y':
        axis_mask = (np.abs(x) < 1e-3) & (np.abs(z) < 1e-3)
        y_axis = y[axis_mask]
        g_axis = Bz[axis_mask]
        
        coeffs = np.polyfit(y_axis, g_axis, 1)
        slope = coeffs[0]

        print("\n===== Y GRADIENT PERFORMANCE =====")
        print(f"Gradient efficiency (mT/m/A): {slope*1e3:.3f}")

        linear_fit = np.polyval(coeffs, y_axis)
        error = (g_axis - linear_fit) / np.max(np.abs(linear_fit)) * 100
        print(f"Max linearity error within DSV (%): {np.max(np.abs(error)):.2f}")

    elif axis == 'z':
        axis_mask = (np.abs(x) < 1e-3) & (np.abs(y) < 1e-3)
        z_axis = z[axis_mask]
        g_axis = Bz[axis_mask]
        
        coeffs = np.polyfit(z_axis, g_axis, 1)
        slope = coeffs[0]

        print("\n===== Z GRADIENT PERFORMANCE =====")
        print(f"Gradient efficiency (mT/m/A): {slope*1e3:.3f}")

        linear_fit = np.polyval(coeffs, z_axis)
        error = (g_axis - linear_fit) / np.max(np.abs(linear_fit)) * 100
        print(f"Max linearity error within DSV (%): {np.max(np.abs(error)):.2f}")


# ===============================
# CHECK SPHERICAL HARMONICS
#================================


def run_spherical_harmonic_diagnostic(result):

    coords = result.target_field.coords
    field = result.coil_gradient.gradient_in_target_direction

    x,y,z = coords

    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y,x)

    max_l = 3

    coeffs = {}

    for l in range(max_l+1):
        for m in range(-l,l+1):

            Ylm = sph_harm_y(l,m,phi,theta).real
            a = np.sum(field*Ylm)/np.sum(Ylm**2)

            coeffs[(l,m)] = a

    print("\n===== Spherical Harmonic Content =====")

    for k,v in coeffs.items():
        print(f"l={k[0]}, m={k[1]} : {v:.4e}")



