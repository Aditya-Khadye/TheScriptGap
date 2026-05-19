from lxml import etree

GITHUB_BASE = "https://raw.githubusercontent.com/Aditya-Khadye/TheScriptGap/refs/heads/main/docs/viz/network_images/Arabic"

def add_image_urls_to_gexf(input_path: str, output_path: str = None) -> None:
    """
    Add image_url attributes to an existing GEXF file without disturbing
    any existing attributes (viz:color, viz:size, viz:position, etc.)

    Args:
        input_path:  Path to the existing .gexf file
        output_path: Path to save the updated file (defaults to overwriting input)
    """
    if output_path is None:
        output_path = input_path

    tree = etree.parse(input_path)
    root = tree.getroot()

    # Detect namespace
    ns = {"gexf": root.nsmap.get(None, "http://www.gexf.net/1.2draft")}
    gexf_ns = ns["gexf"]

    # --- 1. Register the new attribute definition in the node <attributes> block ---
    graph = root.find(f"{{{gexf_ns}}}graph")
    node_attrs_block = None
    for attrs in graph.findall(f"{{{gexf_ns}}}attributes"):
        if attrs.get("class") == "node":
            node_attrs_block = attrs
            break

    # Create the block if it doesn't exist
    if node_attrs_block is None:
        node_attrs_block = etree.SubElement(graph, f"{{{gexf_ns}}}attributes")
        node_attrs_block.set("class", "node")

    # Find the next available attribute id
    existing_ids = [
        int(a.get("id")) for a in node_attrs_block.findall(f"{{{gexf_ns}}}attribute")
        if a.get("id", "").isdigit()
    ]
    new_attr_id = str(max(existing_ids, default=-1) + 1)

    # Check image_url isn't already declared
    already_exists = any(
        a.get("title") == "image_url"
        for a in node_attrs_block.findall(f"{{{gexf_ns}}}attribute")
    )
    if not already_exists:
        new_attr = etree.SubElement(node_attrs_block, f"{{{gexf_ns}}}attribute")
        new_attr.set("id", new_attr_id)
        new_attr.set("title", "image_url")
        new_attr.set("type", "string")

    # --- 2. Add <attvalue> to each node ---
    nodes_el = graph.find(f"{{{gexf_ns}}}nodes")
    updated = 0
    for node in nodes_el.findall(f"{{{gexf_ns}}}node"):
        node_id = node.get("id")
        image_url = f"{GITHUB_BASE}/{node_id}.png"

        attvalues = node.find(f"{{{gexf_ns}}}attvalues")
        if attvalues is None:
            attvalues = etree.SubElement(node, f"{{{gexf_ns}}}attvalues")

        # Skip if already has image_url
        already_set = any(
            av.get("for") == new_attr_id for av in attvalues.findall(f"{{{gexf_ns}}}attvalue")
        )
        if not already_set:
            av = etree.SubElement(attvalues, f"{{{gexf_ns}}}attvalue")
            av.set("for", new_attr_id)
            av.set("value", image_url)
            updated += 1

    tree.write(output_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    print(f"Added image_url to {updated} nodes -> saved to {output_path}")

add_image_urls_to_gexf(r"C:\Users\User\Desktop\TheScriptGap\similarity_research\diversity_research\font_similarity_outputs\full_font_similarity_pairs\network_files\font_network_example.gexf")