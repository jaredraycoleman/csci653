import random
import time
from functools import lru_cache
from hashlib import sha256
from typing import List, Set

import dask
import networkx as nx


def top_down_color_hash(color: str, parent_top_down_color_hashes: Set[str]) -> str:
    """Compute the top-down color hash of a node.
    
    Args:
        color (str): The color of the node
        parent_top_down_color_hashes (Set[str]): The top down color hashes of the parents of the node

    Returns:
        str: The top down color hash of the node
    """
    return sha256((color + "".join(sorted(parent_top_down_color_hashes))).encode()).hexdigest()

def bottom_up_color_hash(color: str, child_bottom_up_color_hashes: Set[str]) -> str:
    """Compute the bottom-up color hash of a node.

    Args:
        color (str): The color of the node
        child_bottom_up_color_hashes (Set[str]): The bottom up color hashes of the children of the node

    Returns:
        str: The bottom up color hash of the node
    """
    return sha256((color + "".join(sorted(child_bottom_up_color_hashes))).encode()).hexdigest()

def color_hash(top_down_color_hash: str, bottom_up_color_hash: str) -> str:
    """Compute the color hash of a node.

    Args:
        top_down_color_hash (str): The top down color hash of the node
        bottom_up_color_hash (str): The bottom up color hash of the node

    Returns:
        str: The color hash of the node
    """
    return sha256((top_down_color_hash + bottom_up_color_hash).encode()).hexdigest()

def set_color_hashes(graph: nx.DiGraph) -> nx.DiGraph:
    """Set the color hashes of the nodes in the graph.

    Args:
        graph (nx.DiGraph): The graph

    Returns:
        nx.DiGraph: The graph with the color hashes set
    """
    graph = graph.copy()
    top_sort = list(nx.topological_sort(graph))
    for node in reversed(top_sort):
        graph.nodes[node]["bottom_up_color_hash"] = bottom_up_color_hash(graph.nodes[node]["color"], {graph.nodes[child]["bottom_up_color_hash"] for child in graph.successors(node)})
    for node in top_sort:
        graph.nodes[node]["top_down_color_hash"] = top_down_color_hash(graph.nodes[node]["color"], {graph.nodes[child]["top_down_color_hash"] for child in graph.predecessors(node)})
        graph.nodes[node]["color_hash"] = color_hash(graph.nodes[node]["top_down_color_hash"], graph.nodes[node]["bottom_up_color_hash"])
    return graph

def rec_set_color_hashes(graph: nx.DiGraph) -> nx.DiGraph:
    """Parallelized divide and conquer algorithm to set the color hashes of the nodes in the graph.

    Args:
        graph (nx.DiGraph): The graph

    Returns:
        nx.DiGraph: The graph with the color hashes set
    """
    @lru_cache(maxsize=None)
    def node_td_hash(node: str) -> str:
        """Compute the top-down color hash of a node.

        Args:
            node (str): The node

        Returns:
            str: The top-down color hash of the node
        """
        return top_down_color_hash(graph.nodes[node]["color"], {node_td_hash(child) for child in graph.successors(node)})

    @lru_cache(maxsize=None)
    def node_bu_hash(node: str) -> str:
        """Compute the bottom-up color hash of a node.

        Args:
            node (str): The node

        Returns:
            str: The bottom-up color hash of the node
        """
        return bottom_up_color_hash(graph.nodes[node]["color"], {node_bu_hash(child) for child in graph.predecessors(node)})

    @lru_cache(maxsize=None)
    def node_hash(node: str) -> str:
        """Compute the color hash of a node.

        Args:
            node (str): The node

        Returns:
            str: The color hash of the node
        """
        return color_hash(node_td_hash(node), node_bu_hash(node))

    for node in graph.nodes:
        graph.nodes[node]["top_down_color_hash"] = node_td_hash(node)
        graph.nodes[node]["bottom_up_color_hash"] = node_bu_hash(node)
        graph.nodes[node]["color_hash"] = node_hash(node)


def random_dag() -> nx.DiGraph:
    """Generate a random DAG.

    Returns:
        nx.DiGraph: The random DAG
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(range(10000))
    for node in graph.nodes:
        graph.nodes[node]["color"] = random.choice(["red", "green", "blue", "yellow", "purple", "orange", "pink", "black", "white"])
    for node in graph.nodes:
        for dst in range(node+1, 100):
            if random.random() < 0.2:
                graph.add_edge(node, dst)
    assert(nx.is_directed_acyclic_graph(graph))
    return graph

def main():
    graph = random_dag()
    print(f"Graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")


if __name__ == "__main__":
    main()

