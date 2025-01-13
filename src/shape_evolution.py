import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
riemann = np.load('/home/pavel/cell_morphology/nov30/riemann_distances.npy', allow_pickle=True)
times = np.load('/home/pavel/cell_morphology/nov30/times.npy', allow_pickle=True)
centr = np.load('/home/pavel/cell_morphology/nov30/centroids.npy', allow_pickle=True)
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

# for i in range(len(centr)): 
#     x_coords = [c[0] for c in centr[i]]
#     y_coords = [c[1] for c in centr[i]]
#     print(x_coords)
#     x_smooth = gaussian_filter1d(x_coords, sigma=2)
#     y_smooth = gaussian_filter1d(y_coords, sigma=2)
    
#     smooth_centr = [[x, y] for x, y in zip(x_smooth, y_smooth)]
#     centr[i] = smooth_centr




def plot_cell_trajectory(n):
    centroids = centr[n - 1]  
    centroids = np.array(centroids)
    x_coords = centroids[:, 0]  
    y_coords = centroids[:, 1] 

    # riemann_data = riemann[n-1] 

    # riemann_data = np.array(riemann_data)  

    time_steps = np.arange(len(x_coords))  

    plt.figure(figsize=(10, 8))
    plt.scatter(
        x_coords[0], y_coords[0],  
        c='black',
        marker='o',
        edgecolor='k',
        s=100,
        alpha=0.7,
        label='Start Point'
    )
    scatter = plt.scatter(
        x_coords[1:], y_coords[1:],  
        c=time_steps[1:],#riemann_data[1:],           
        cmap='plasma',             
        marker='o',
        edgecolor='k',
        s=100,
        alpha=0.7
    )
    plt.plot(x_coords, y_coords, linestyle='-', color='gray', alpha=0.5)  # Линия траектории

    plt.title(f"Cell Num {n}")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Time Step (t)', rotation=270, labelpad=15)

    plt.show()



import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def save_all_cells_to_html_grid_with_legend(cell_indices, cell_dir='cells', output_file='all_cells_visualization.html'):

    valid_cells = []

    for n in cell_indices:
       
        sorted_cells = sorted(os.listdir(cell_dir), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
        if n - 1 >= len(sorted_cells):
            print(f"Error: Cell {n} is out of range. Skipping...")
            continue
        cell_path = os.path.join(cell_dir, sorted_cells[n - 1])

        cell_frames = sorted(os.listdir(cell_path), key=lambda x: int(''.join(filter(str.isdigit, x))))  # Correct sorting
        if len(cell_frames) < 15:
            print(f"Skipping cell {n} because it has less than 15 frames.")
            continue

        valid_cells.append(n)

    # Create make_subplots only for valid cells
    n_rows = len(valid_cells)
    n_cols = 2  # 3D plot + Riemann distance plot
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=[[{'type': 'scene'}, {'type': 'xy'}] for _ in range(n_rows)],
        subplot_titles=[
            f"Cell {cell} (3D)" if col == 0 else f"Riemann Distances (Cell {cell})"
            for cell in valid_cells for col in range(n_cols)
        ]
    )

    for idx, n in enumerate(valid_cells):
        row = idx + 1

        # Path to the selected cell folder
        sorted_cells = sorted(os.listdir(cell_dir), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
        cell_path = os.path.join(cell_dir, sorted_cells[n - 1])

        # Sort cell frames numerically
        cell_frames = sorted(os.listdir(cell_path), key=lambda x: int(''.join(filter(str.isdigit, x))))  # Sort frames by number

        print(f"Processing cell {n} data from: {cell_path}")

        # Prepare Riemann distances
        riemann_values = riemann[n - 1]
        times = np.arange(len(riemann_values))  # Assuming times are evenly spaced from 0 to N-1

        # Normalize Riemann distances
        cmap = plt.cm.plasma
        norm = plt.Normalize(vmin=min(riemann_values), vmax=max(riemann_values))

        # Add 3D plot for the cell
        for i, frame in enumerate(cell_frames):
            frame_path = os.path.join(cell_path, frame)

            # Load data
            time = np.load(os.path.join(frame_path, 'time.npy'))
            outline = np.load(os.path.join(frame_path, 'outline.npy'))

            # Bind color to the Riemann distance
            if i == 0:
                color = 'rgba(0, 0, 0, 1)'  # Initial frame is black
                linewidth = 2
            else:
                rgba_color = cmap(norm(riemann_values[i - 1]))
                color = f"rgba({rgba_color[0]*255}, {rgba_color[1]*255}, {rgba_color[2]*255}, {rgba_color[3]})"
                linewidth = 1

            # Add the line to the 3D subplot
            fig.add_trace(go.Scatter3d(
                x=outline[:, 0],
                y=outline[:, 1],
                z=np.full(len(outline[:, 1]), time),  # Time axis
                mode='lines',
                line=dict(color=color, width=linewidth),
                name=f"Frame {i + 1} (Cell {n})" if i > 0 else f"Start Frame (Cell {n})",
                showlegend=False
            ), row=row, col=1)

        # Add scatter plot for Riemann distances
        fig.add_trace(go.Scatter(
            x=times,
            y=riemann_values,
            mode='markers',  # Only points, no lines
            marker=dict(size=8, color='blue'),
            name=f"Riemann Distances (Cell {n})"
        ), row=row, col=2)

    # Adjust layout size
    fig.update_layout(
        title="3D Visualizations of Cells with Riemann Distances",
        height=1200 * n_rows,  # Increase height depending on the number of rows
        width=2000,           # Width for two columns
        margin=dict(
            l=200,  # Left margin
            r=200,  # Right margin
            t=100,  # Top margin
            b=100   # Bottom margin
        ),
    )

    # Save the figure to HTML
    fig.write_html(output_file)
    print(f"All cells visualizations have been saved to {output_file}. Open the file in a browser to view it.")

def plot_all_cells_xz(cell_dir="cells/", output_dir = "pic/", cells_per_page=20):

    
    cells = sorted(os.listdir(cell_dir))
    num_cells = len(cells)
    num_pages = (num_cells + cells_per_page - 1) // cells_per_page  
    
    for page in range(num_pages):
        fig, axes = plt.subplots(int(cells_per_page/5), 5, figsize=(5 * cells_per_page, 5 * cells_per_page))
        axes = np.array(axes).flatten() 
        
        start = page * cells_per_page
        end = min(start + cells_per_page, num_cells)
        
        for i, cell_idx in enumerate(range(start, end)):
            cell_name = cells[cell_idx]
            cell_path = os.path.join(cell_dir, cell_name)
            frames = sorted(
                os.listdir(cell_path), 
                key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else x
            )

            ax = axes[i]
            cmap = plt.cm.plasma
            colors = cmap(np.linspace(0, 1, len(frames)))
            
            for frame_idx, frame in enumerate(frames):
                frame_path = os.path.join(cell_path, frame)
                time = np.load(os.path.join(frame_path, 'time.npy'))
                outline = np.load(os.path.join(frame_path, 'outline.npy'))
                
                ax.plot(outline[:, 0], time * np.ones(len(outline[:, 0])), '-', c=colors[frame_idx])
            
            ax.set_title(f'Cell {cell_idx + 1} - XZ Projection')
            ax.set_xlabel('X Coordinate')
            ax.legend(fontsize=8, loc='upper right')
            ax.set_ylabel('Time (Z Coordinate)')
            ax.grid(True)
        
        for j in range(i + 1, cells_per_page):
            fig.delaxes(axes[j])
        
        output_path = os.path.join(output_dir, f'xz_projections_page_{page + 1}.png')
        fig.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)

def plot_cell(n):
    cell_dir = 'cells'
    cell_path = os.path.join(
        cell_dir, 
        sorted(os.listdir(cell_dir), key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)[n-1]
    )

    cell = sorted(
        os.listdir(cell_path), 
        key=lambda x: int(''.join(filter(str.isdigit, x))) 
    )
    print(cell)



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.cm.plasma  
    riemann_values = riemann[n-1]  
    print(riemann_values)
    norm = plt.Normalize(vmin=min(riemann_values[1:]), vmax=max(riemann_values[1:]))

    for i, frame in enumerate(cell):
        if(i == 0):
            continue
        frame_path = os.path.join(cell_path, frame)
    
        time = np.load(os.path.join(frame_path, 'time.npy'))
        outline = np.load(os.path.join(frame_path, 'outline.npy'))
        centroid = np.load(os.path.join(frame_path, 'centroid.npy'))
        
 
        if i == 0:
            color = 'black'  
            linewidth = 2
        else:
            color = cmap(norm(riemann_values[i]))  
            linewidth = 1

     
        ax.plot3D(
            outline[:, 0], 
            outline[:, 1], 
            np.full(len(outline[:, 1]), time),
            color=color, 
            linewidth=linewidth
        )
        print(f"Frame {frame}: Time = {time}, Riemann Distance = {riemann_values[i]}")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, orientation='vertical')
    cbar.set_label('Riemann Distance', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Time')
    
    plt.show()
def plot_cell_by_motion_type(n):

    cell_dir = 'cells'
    cell_path = os.path.join(
        cell_dir, 
        sorted(os.listdir(cell_dir), key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)[n-1]
    )

    cell = sorted(
        os.listdir(cell_path), 
        key=lambda x: int(''.join(filter(str.isdigit, x))) 
    )
    print(cell)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    interval_colors = {
        0: "brown", 
        1: "blue",       
        2: "cyan",       
        3: "magenta",  
        "unclassified": "black"  
    }
    import h5py

    with h5py.File('time_events_95.h5', 'r') as f:
        track_i_data = f[f'/track_{n}'][:]
        first_three_rows = track_i_data[:3]
        event_indices = first_three_rows[0, :].astype(int) - 1
        interval_types = first_three_rows[2, :]  

    for i, frame in enumerate(cell):
        frame_path = os.path.join(cell_path, frame)
        time = np.load(os.path.join(frame_path, 'time.npy'))
        outline = np.load(os.path.join(frame_path, 'outline.npy'))

        current_type = None
        for start_idx, interval_type in enumerate(interval_types):
            if i >= event_indices[start_idx] and (start_idx + 1 == len(event_indices) or i < event_indices[start_idx + 1]):
                current_type = interval_type
                break

        if current_type is not None:
            interval_type = int(current_type) if not np.isnan(current_type) else "unclassified"
            color = interval_colors.get(interval_type, "black")

        ax.plot3D(
            outline[:, 0], 
            outline[:, 1], 
            np.full(len(outline[:, 1]), time),
            color=color, 
            linewidth=1
        )
        print(f"Frame {frame}: Time = {time}, Motion Type = {current_type}")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="brown", lw=2, label="Immobile"),
        Line2D([0], [0], color="blue", lw=2, label="Confined Diffusion"),
        Line2D([0], [0], color="cyan", lw=2, label="Free Diffusion"),
        Line2D([0], [0], color="magenta", lw=2, label="Directed Diffusion"),
        Line2D([0], [0], color="black", lw=2, label="Unclassified")
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Time')
    ax.set_title(f"Cell {n}")
    
    plt.show()


if __name__ == "__main__":
    #plot_all_cells_xz()
    #plot_cell(6)
    plot_cell_by_motion_type(87)
    #plot_cell_trajectory(87)
    # save_cell_to_html(
    # n=1,  #
    # cell_dir='cells',
    # output_file='cell_visualization.html'
#)
    #  save_all_cells_to_html_grid_with_legend(
    #      cell_indices=list(range(1, 10)),  # Индексы клеток от 1 до 204s
    #      cell_dir='cells',
    #      output_file='all_cells_visualization_with_legend.html',
    #  )