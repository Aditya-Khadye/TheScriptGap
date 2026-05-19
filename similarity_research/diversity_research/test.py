import networkx as nx

GITHUB_BASE = "https://raw.githubusercontent.com/Aditya-Khadye/TheScriptGap/refs/heads/main/docs/viz/network_images"

def add_image_urls_to_gexf(input_path: str, output_path: str = None) -> nx.Graph:
    """
    Load an existing GEXF file and add image_url attributes to all nodes.
    
    Args:
        input_path:  Path to the existing .gexf file
        output_path: Path to save the updated file (defaults to overwriting input)
    """
    if output_path is None:
        output_path = input_path

    G = nx.read_gexf(input_path)

    for node in G.nodes():
        G.nodes[node]["image_url"] = f"{GITHUB_BASE}/Arabic/{node}.png"

    nx.write_gexf(G, output_path)
    print(f"Updated {G.number_of_nodes()} nodes -> saved to {output_path}")
    
    return G


add_image_urls_to_gexf(r"C:\Users\User\Desktop\TheScriptGap\similarity_research\diversity_research\font_similarity_outputs\full_font_similarity_pairs\network_files\font_network_example.gexf")