import os
import glob
import pandas as pd
import networkx as nx
import re
import math

def load_data(experiment_path):
    """
    Loads all fragment properties CSVs from an experiment folder into a single DataFrame.
    """
    csv_dir = os.path.join(experiment_path, 'csv')
    csv_files = glob.glob(os.path.join(csv_dir, 'fragments_properties_step_*.csv'))
    
    if not csv_files:
        print(f"Error: No 'fragments_properties_step_*.csv' files found in '{csv_dir}'")
        return None

    all_fragments_df = []
    # Sort files numerically by step number to ensure correct processing order
    try:
        csv_files.sort(key=lambda f: int(re.search(r'step_(\d+)\.csv', f).group(1)))
    except AttributeError:
        print("Error: Could not determine step number from CSV filenames. Ensure they are named like '...step_1.csv'.")
        return None

    for f in csv_files:
        try:
            match = re.search(r'step_(\d+)\.csv', os.path.basename(f))
            if match:
                step = int(match.group(1))
                df = pd.read_csv(f)
                df['step'] = step
                all_fragments_df.append(df)
        except Exception as e:
            print(f"Warning: Could not read or process file {f}. Error: {e}")

    if not all_fragments_df:
        print("Error: Failed to load any fragment data.")
        return None

    full_df = pd.concat(all_fragments_df, ignore_index=True)
    full_df['parent_id'] = full_df['parent_id'].astype(int)
    return full_df

def build_genealogy_graph(df):
    """
    Builds a directed graph of fragment evolution using networkx.
    """
    G = nx.DiGraph()
    
    for _, row in df.iterrows():
        node_id = (row['step'], row['fragment_id'])
        G.add_node(node_id, area=row['area'], num_balls=row['num_balls'])

    for _, row in df.iterrows():
        if row['parent_id'] != -1:
            parent_node_id = (row['step'] - 1, row['parent_id'])
            child_node_id = (row['step'], row['fragment_id'])
            if parent_node_id in G:
                G.add_edge(parent_node_id, child_node_id)
                
    return G

def plot_full_tree(graph, output_filename):
    """
    Plots the entire fragment evolution graph using Graphviz.
    """
    try:
        import graphviz
    except ImportError:
        print("\nERROR: Plotting requires the 'graphviz' library.")
        print("Please install it by running: pip install graphviz")
        print("You also need to install the Graphviz software from https://graphviz.org/download/")
        return

    dot = graphviz.Digraph(comment='Fragment Evolution')
    dot.attr(rankdir='TB', splines='spline', overlap='false')

    # Find the maximum area for normalization
    max_area = max(graph.nodes[n].get('area', 0) for n in graph.nodes) if graph.nodes else 1

    # Group nodes by step (rank) for a clean top-to-bottom layout
    for step in sorted(list(set(n[0] for n in graph.nodes))):
        with dot.subgraph() as s:
            s.attr(rank='same')
            for node in [n for n in graph.nodes if n[0] == step]:
                area = graph.nodes[node].get('area', 0)
                
                # Style node based on its area
                # Normalize area for color calculation (log scale)
                norm_area = math.log1p(area) / math.log1p(max_area) if max_area > 1 else 0
                
                # Interpolate from blue (small) to red (large)
                red = int(255 * norm_area)
                blue = int(255 * (1 - norm_area))
                fillcolor = f'#{red:02x}64{blue:02x}' # R-G-B format

                s.node(
                    name=str(node),
                    label=f"S{node[0]}\nID:{node[1]}\n{area:.1f}",
                    shape='box',
                    style='filled',
                    fillcolor=fillcolor,
                    fontcolor='white' if norm_area > 0.5 else 'black',
                    fontsize='8',
                    height='0.4',
                    width='0.6'
                )

    # Add all edges
    for edge in graph.edges:
        dot.edge(str(edge[0]), str(edge[1]))

    print(f"\nRendering full evolution tree to '{output_filename}'...")
    try:
        output_format = os.path.splitext(output_filename)[1][1:] or 'png'
        dot.render(os.path.splitext(output_filename)[0], format=output_format, view=False, cleanup=True)
        print(f"Success! Image saved to '{output_filename}'")
    except graphviz.backend.ExecutableNotFound:
        print("\nERROR: Graphviz executable not found.")
        print("Please ensure the Graphviz software is installed and in your system's PATH.")
        print("Download from: https://graphviz.org/download/")
    except Exception as e:
        print(f"An error occurred during rendering: {e}")

def main():
    """
    Main function to run the analysis script.
    """
    print("Fragment Evolution Analysis Tool")
    print("=" * 30)

    experiments = [d for d in glob.glob('experiments/*') if os.path.isdir(d)]
    if not experiments:
        print("Error: No experiment folders found in the 'experiments' directory.")
        return

    print("Available experiments:")
    for i, exp in enumerate(experiments):
        print(f"  {i+1}: {os.path.basename(exp)}")
    
    try:
        choice = int(input(f"Select an experiment to analyze (1-{len(experiments)}): ")) - 1
        if not 0 <= choice < len(experiments):
            raise ValueError()
        experiment_path = experiments[choice]
    except (ValueError, IndexError):
        print("Invalid selection. Exiting.")
        return

    print(f"\nLoading data for '{os.path.basename(experiment_path)}'...")
    df = load_data(experiment_path)
    
    if df is None:
        return

    print("Building genealogy graph...")
    graph = build_genealogy_graph(df)
    print(f"Graph built with {graph.number_of_nodes()} fragments and {graph.number_of_edges()} connections.")

    exp_name = os.path.basename(experiment_path)
    output_filename = f"evolution_tree_{exp_name}.png"

    plot_full_tree(graph, output_filename)

if __name__ == "__main__":
    # Advise user on dependencies
    try:
        import pandas as pd
        import networkx as nx
    except ImportError:
        print("="*50)
        print("IMPORTANT: This script requires the 'pandas' and 'networkx' libraries.")
        print("Please install them by running:")
        print("pip install pandas networkx")
        print("="*50)
    else:
        main()
