import matplotlib.pyplot as plt
import networkx as nx

def draw_mind_map(data: dict, title: str = "Mind Map"):
    G = nx.Graph()

    def add_edges(parent, children):
        for child in children:
            if isinstance(child, dict):
                for sub_parent, sub_children in child.items():
                    G.add_edge(parent, sub_parent)
                    add_edges(sub_parent, sub_children)
            else:
                G.add_edge(parent, child)

    for root, branches in data.items():
        add_edges(root, branches)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 8))
    nx.draw(
        G, pos, with_labels=True, node_color="skyblue", node_size=2000,
        font_size=10, font_weight='bold', edge_color="gray"
    )
    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.savefig(title.replace(" ", "_").lower() + ".png")
    plt.close()