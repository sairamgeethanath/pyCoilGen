import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import trimesh
from scipy.interpolate import splprep, splev
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow
import pyvista as pv

# ----------------------------
# debug plot
# ----------------------------
def plot_wire_loops(wire_loops):
    fig, ax = plt.subplots(figsize=(6,6))
    for loop in wire_loops:
        p = loop["path"]
        ax.plot(p[:,0], p[:,1], 'r' if loop["sign"]>0 else 'b')
    ax.set_aspect("equal")
    ax.set_title("Gradient Coil Wire Paths")
    plt.show()

def plot_wire_loops_tube(wire_loops, wire_width):

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


def plot_gerber_paths(gerber_paths, plate_radius_mm, wire_width_mm, part_index):
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

def plot_stl_patch(gerber_paths, plate_radius_mm, wire_width_mm, part_index):
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