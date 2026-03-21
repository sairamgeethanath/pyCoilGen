
import numpy as np
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
import pyvista as pv
from pyCoilGen.plotting.plot_wire_loops import plot_wire_loops, plot_wire_loops_tube


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
        plot_wire_loops(wire_loops)
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
       plot_wire_loops_tube(wire_loops, wire_width)

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