import json
import pathlib
from functools import lru_cache
from hashlib import sha256
from typing import Dict, Generator, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt


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


def load_dag_from_json(path: Union[pathlib.Path, str]) -> nx.DiGraph:
    path = pathlib.Path(path)
    data = json.loads(path.read_text())
    graph = nx.DiGraph()

    for task in data['workflow']['tasks']:
        graph.add_node(task['name'], color=task['category'])
    
    for task in data['workflow']['tasks']:
        for parent in task['parents']:
            graph.add_edge(parent, task['name'])
    
    return graph

def load_dags_from_json(path: Union[pathlib.Path, str]) -> List[nx.DiGraph]:
    return [
        load_dag_from_json(json_path)
        for json_path in pathlib.Path(path).glob("*.json")
    ]

def draw_workflow(workflow: nx.DiGraph, 
                  color_attr: Optional[str] = None,
                  ax: Optional[plt.Axes] = None) -> None:
    """Draw workflow DAG
    
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    if color_attr is not None:
        color_idx = {color: i for i, color in enumerate(sorted(set(nx.get_node_attributes(workflow, color_attr).values())))}
        cmap = cm.get_cmap('rainbow', len(color_idx))
        node_color = [cmap(color_idx[workflow.nodes[node][color_attr]]) for node in workflow.nodes]
    else:
        node_color = ['black' for task in workflow.nodes]
    pos = nx.nx_agraph.pygraphviz_layout(workflow, prog='dot')
    nx.draw(
        workflow, pos, ax=ax, 
        with_labels=False, 
        node_color=node_color
    )
    if color_attr is not None:
        for col, i in color_idx.items():
            plt.scatter([],[], c=[cmap(i)], label=col)
    return ax.get_figure(), ax

def get_root_idx(paths: List[List[str]]) -> int:
    """Get the index of the root node in the paths.

    Args:
        paths (List[List[str]]): The paths

    Returns:
        int: The index of the root node in the paths
    """
    min_len = min(len(path) for path in paths)
    arr = np.array([path[:min_len] for path in paths])
    return np.argmax(np.all(arr == arr[0, :], axis=0))

def get_patterns(paths: List[List[str]]) -> Generator[List[str], None, None]:
    """Get the patterns from the paths.

    Args:
        paths (List[List[str]]): The paths

    Yields:
        List[str]: The patterns
    """
    root_idx = get_root_idx(paths)
    for i in range(len(paths)):
        yield paths[i][root_idx:]
        paths[i] = paths[i][root_idx+1:] # trim path

    # group paths by commmon first element
    path_groups: Dict[str, List[str]] = {}
    for path in paths:
        if len(path) > 0:
            path_groups.setdefault(path[0], []).append(path)
    
    # recurse on each group
    for group in path_groups.values():
        if len(group) > 1:
            yield from get_patterns(group)
    
def task_func(task: str, color_hash: str, in_messages: List[Tuple[str, List[str]]]) -> Tuple[List[str], Tuple[str, List[str]]]:
    """

    Args:
        in_messages (List[Tuple[str, List[str]]]): List of tuples of the form (color_hash, [critical_path])
    
    Returns:
        Tuple[str, List[str]]: Tuple of the form (color_hash, [critical_path])
    """
    # group messages by color hash
    color_hash_groups = {}
    for color_hash, critical_path in in_messages:
        if color_hash not in color_hash_groups:
            color_hash_groups[color_hash] = []
        color_hash_groups[color_hash].append(critical_path)

    patterns = []
    for color_hash, critical_paths in color_hash_groups.items():
        # get the patterns from the critical paths
        if len(critical_paths) > 1:
            patterns.extend(get_patterns(critical_paths))
    
    paths = [path for _, path in in_messages]
    root_idx = get_root_idx(paths)
    critical_path = [*paths[0][:root_idx+1], task]
    return patterns, (color_hash, critical_path)

def find_patterns(graph: nx.DiGraph):
    # compute color hashes
    set_color_hashes(graph)
    top_sort = list(nx.topological_sort(graph))
    messages = {}
    for node in top_sort:
        patterns, msg = task_func(node, graph.nodes[node]["color_hash"], messages.get(node, [('SRC', ['SRC'])]))
        for child in graph.successors(node):
            if child not in messages:
                messages[child] = []
            messages[child].append(msg)
        for pattern in patterns:
            print(f"Pattern: {pattern}")

thisdir = pathlib.Path(__file__).parent.resolve()
def main():
    graph = load_dag_from_json(thisdir.joinpath('pegasus-instances', 'montage', 'chameleon-cloud', 'montage-chameleon-2mass-01d-001.json'))
    graph = set_color_hashes(graph)
    find_patterns(graph)

    fig, ax = draw_workflow(graph, color_attr="color_hash")
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()

