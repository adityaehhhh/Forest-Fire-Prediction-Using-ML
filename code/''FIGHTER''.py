import networkx as nx
import pandas as pd
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import os

def load_dataset(nodes=100):
    if os.path.exists('./covtype.data'):
        df = pd.read_csv('./covtype.data', header=None)
        if len(df) < nodes:
            raise ValueError(f"Dataset has fewer than {nodes} rows.")
        return df.iloc[:nodes][[0, 1, 2]] 
    else:
        print("Dataset not found. Creating sample dataset.")
        return pd.DataFrame({
            0: np.random.randint(1800, 4000, nodes),  
            1: np.random.randint(0, 60, nodes),     
            2: np.random.randint(0, 360, nodes)      
        })

def node_threshold(slope, elevation, ele_min, ele_max, aspect):
    phi = np.tan(np.radians(slope))
    phi_s = 5.275 * phi**2
    h = (elevation - ele_min) / (ele_max - ele_min) * 2300
    h_prime = max(h * np.exp(-6), 1)
    xi = 1 / (1 + np.log(h_prime))
    alpha = 0.2  
    theta = -np.arctan(phi_s * xi * alpha) / np.pi + 0.5
    return max(0.1, round(theta, 2))  

def dist(pair1, pair2):
    return np.sqrt((pair2[0] - pair1[0])**2 + (pair2[1] - pair1[1])**2)

def initialize_grid_graph(nodes, tree_density):
    g = nx.Graph()
    grid_size = int(np.sqrt(nodes))
    pos_dict = {}
    colors = ['green'] * nodes if tree_density == 1.0 else ['green' if rnd.random() < tree_density else 'black' for _ in range(nodes)]
    empty_list = [i + 1 for i in range(nodes) if colors[i] == 'black']

    df = load_dataset(nodes)
    for i in range(grid_size):
        for j in range(grid_size):
            k = i * grid_size + j + 1
            if k > nodes:
                break
            elevation = df.iloc[k-1, 0]
            slope = df.iloc[k-1, 1]
            aspect = df.iloc[k-1, 2]
            theta = node_threshold(slope, elevation, df[0].min(), df[0].max(), aspect)
            lf = rnd.randint(15, 25)  
            pos = (j * 1.0, (grid_size - 1 - i) * 1.0)  
            fire_state = 'not_burnt' if colors[k-1] == 'green' else 'empty'
            g.add_node(k, threshold_switch=theta, color=colors[k-1], fire_state=fire_state, life=lf, pos=pos)
            pos_dict[k] = pos

    for i in range(grid_size):
        for j in range(grid_size):
            k = i * grid_size + j + 1
            if k <= nodes:
                # Right neighbor
                if j < grid_size - 1:
                    k_right = k + 1
                    if g.nodes[k]['fire_state'] != 'empty' and k_right in g.nodes and g.nodes[k_right]['fire_state'] != 'empty':
                        g.add_edge(k, k_right, w=1.0, color='green')
                # Down neighbor
                if i < grid_size - 1:
                    k_down = k + grid_size
                    if k_down <= nodes and g.nodes[k]['fire_state'] != 'empty' and k_down in g.nodes and g.nodes[k_down]['fire_state'] != 'empty':
                        g.add_edge(k, k_down, w=1.0, color='green')
                # Left neighbor
                if j > 0:
                    k_left = k - 1
                    if g.nodes[k]['fire_state'] != 'empty' and k_left in g.nodes and g.nodes[k_left]['fire_state'] != 'empty':
                        g.add_edge(k, k_left, w=1.0, color='green')
                # Up neighbor
                if i > 0:
                    k_up = k - grid_size
                    if k_up > 0 and g.nodes[k]['fire_state'] != 'empty' and k_up in g.nodes and g.nodes[k_up]['fire_state'] != 'empty':
                        g.add_edge(k, k_up, w=1.0, color='green')

    print(f"Total edges created: {g.number_of_edges()}")
    return g, colors, pos_dict

def simulate_fire_spread(g, colors, pos_dict, wind_speed, wind_direction):
    global timestep, ax, stopped
    non_empty_count = sum(1 for n in g.nodes if g.nodes[n]['fire_state'] != 'empty')
    print(f"Initial non-empty nodes: {non_empty_count}")
    
    ignition_node = rnd.randint(1, nodes)
    while g.nodes[ignition_node]['fire_state'] != 'not_burnt':
        ignition_node = rnd.randint(1, nodes)
    g.nodes[ignition_node]['fire_state'] = 'burning'
    colors[ignition_node-1] = 'orange'
    g.nodes[ignition_node]['color'] = 'orange'

    timesteps = float('inf')  
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.subplots_adjust(bottom=0.2)  

    def next_step(event):
        global timestep
        if not stopped:
            timestep += 1
            update_plot()

    def stop_simulation(event):
        global stopped
        stopped = True
        plt.close()

    ax_next = plt.axes([0.3, 0.05, 0.2, 0.075])
    btn_next = widgets.Button(ax_next, 'Next')
    btn_next.on_clicked(next_step)

    ax_stop = plt.axes([0.6, 0.05, 0.2, 0.075])
    btn_stop = widgets.Button(ax_stop, 'Stop')
    btn_stop.on_clicked(stop_simulation)

    stopped = False
    def update_plot():
        if not stopped:
            burning_nodes = [n for n in g.nodes if g.nodes[n]['fire_state'] == 'burning']
            if burning_nodes:
                print(f"Timestep {timestep}, Burning nodes: {len(burning_nodes)}")
                for node in burning_nodes:
                    for nb in g.neighbors(node):
                        if g.nodes[nb]['fire_state'] == 'not_burnt':
                            pos_node = g.nodes[node]['pos']
                            pos_nb = g.nodes[nb]['pos']
                            dx = pos_nb[0] - pos_node[0]
                            dy = pos_nb[1] - pos_node[1]
                            angle_to_nb = np.degrees(np.arctan2(dy, dx)) % 360
                            angle_diff = min(abs(wind_direction - angle_to_nb), 360 - abs(wind_direction - angle_to_nb))
                            wind_factor = 1 + (wind_speed / 2) * np.cos(np.radians(angle_diff))  
                            base_prob = g.nodes[nb]['threshold_switch']
                            directional_boost = 2.0 if angle_diff < 30 else 0.5  
                            spread_prob = max(0.4, base_prob * wind_factor * directional_boost)  
                            print(f"Node {nb}, Angle Diff: {angle_diff}, Spread Prob: {spread_prob}, Random: {rnd.random()}")
                            if rnd.random() < spread_prob:
                                g.nodes[nb]['fire_state'] = 'burning'
                                colors[nb-1] = 'orange'
                                g.nodes[nb]['color'] = 'orange'
            
            for node in g.nodes:
                if g.nodes[node]['fire_state'] == 'burning' and g.nodes[node]['life'] > 0:
                    g.nodes[node]['life'] -= 1
                elif g.nodes[node]['fire_state'] == 'burning' and g.nodes[node]['life'] <= 0:
                    g.nodes[node]['fire_state'] = 'burnt'
                    colors[node-1] = 'brown'
                    g.nodes[node]['color'] = 'brown'

            burning = sum(1 for n in g.nodes if g.nodes[n]['fire_state'] == 'burning')
            burnt = sum(1 for n in g.nodes if g.nodes[n]['fire_state'] == 'burnt')
            saved = sum(1 for n in g.nodes if g.nodes[n]['fire_state'] == 'not_burnt')
            
            ax.clear()
            nx.draw(g, ax=ax, node_size=200, node_color=colors, with_labels=False, pos=pos_dict, edge_color='gray')
            ax.set_title(f"Forest Fire Spread - Timestep {timestep}")
            ax.text(0.5, -0.1, f"Burning = {burning}, Burnt = {burnt}, Saved = {saved}", 
                    transform=ax.transAxes, ha='center')
            ax.set_xlim(-1, 11)
            ax.set_ylim(-1, 11)
            fig.canvas.draw()

    update_plot()
    plt.show()

    while not stopped:
        plt.pause(0.1)
    burning = sum(1 for n in g.nodes if g.nodes[n]['fire_state'] == 'burning')
    burnt = sum(1 for n in g.nodes if g.nodes[n]['fire_state'] == 'burnt')
    saved = sum(1 for n in g.nodes if g.nodes[n]['fire_state'] == 'not_burnt')
    print(f"Simulation stopped at Timestep {timestep}")
    print(f"Final Statistics - Burnt: {burnt}, Burning: {burning}, Remaining Trees: {saved}")

nodes = 100  
tree_density = float(input("Enter tree density (0 to 1, e.g., 0.8 for 80% trees): ") or 0.8)
wind_speed = float(input("Enter wind speed (0 to 50, e.g., 10): ") or 25)
wind_direction = float(input("Enter wind direction (0 to 360 degrees, e.g., 90 for East): ") or 90)
timestep = 0

g, colors, pos_dict = initialize_grid_graph(nodes, tree_density)
simulate_fire_spread(g, colors, pos_dict, wind_speed, wind_direction)