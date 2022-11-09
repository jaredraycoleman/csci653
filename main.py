import json
import pathlib
from functools import lru_cache
from hashlib import sha256
from typing import Dict, List, Optional, Set, Tuple, Union
import uuid

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

def full_color_hash(top_down_color_hash: str, bottom_up_color_hash: str) -> str:
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
        graph.nodes[node]["color_hash"] = full_color_hash(graph.nodes[node]["top_down_color_hash"], graph.nodes[node]["bottom_up_color_hash"])
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
        return full_color_hash(node_td_hash(node), node_bu_hash(node))

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

    # Remove forward edges
    _graph = nx.transitive_reduction(graph)
    graph = nx.edge_subgraph(graph, _graph.edges)
    
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

class Signal:
    def __init__(self, signal_class: str, ids: List[int], total_ids: int, node_history: Set["Node"] = set()) -> None:
        self.signal_class = signal_class
        self.ids = ids
        self.total_ids = total_ids
        self.node_history = node_history

    def add_node(self, node: "Node") -> None:
        self.node_history.add(node)

    def __repr__(self) -> str:
        return f"Signal {self.signal_class[:8]} (ids={self.ids} - {len(self.ids)}/{self.total_ids})" #\nNodes: {list(self.node_history)}"
    
    @classmethod
    def combine_signals(self, signals: List["Signal"]) -> "Signal":
        # all color hashes should be the same
        assert len(set(signal.signal_class for signal in signals)) == 1
        signal_class = signals[0].signal_class

        # total ids should be the same
        assert len(set(signal.total_ids for signal in signals)) == 1
        total_ids = signals[0].total_ids

        # combine ids
        ids = {id for signal in signals for id in signal.ids}
        
        # combine node history
        node_history = {node for signal in signals for node in signal.node_history}

        return Signal(
            signal_class=signal_class,
            ids=ids,
            total_ids=total_ids,
            node_history=node_history
        )

class Node:
    def __init__(self, node_id: str, color_hash: str, children: List["Node"]) -> None:
        self.node_id = node_id
        self.color_hash = color_hash
        self.children = children

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __repr__(self) -> str:
        return f"Node {self.node_id} (color_hash={self.color_hash[:8]})"

    def process_signals(self, signals: List[Signal]) -> Tuple[Dict[str, List[Signal]],Dict[str, List[Signal]]]:
        """Process signals at this node.

        Args:
            signals (List[Signal]): The signals to process

        Returns:
            Dict[str, List[Signal]]: output signals. The key is child node id the value is the list of signals.
        """
        # group signals by signal class, then ids
        signals_by_class_and_id: Dict[str, Dict[Tuple[int], List[Signal]]] = {}
        for signal in signals:
            signals_by_class_and_id.setdefault(signal.signal_class, {}).setdefault(tuple(sorted(signal.ids)), []).append(signal)

        # Merge signals with the same ids
        signals_by_class = {
            signal_class: [
                Signal.combine_signals(id_signals)
                for id_signals in signals_by_id.values()
            ]
            for signal_class, signals_by_id in signals_by_class_and_id.items()
        }
        
        # process signals by signal class
        patterns = {}
        output_signals: List[Signal] = []
        for signal_class, signals in signals_by_class.items():
            if len(signals) > 1:
                patterns[signal_class] = signals
                # print(f"Pattern Detected for {signal_class}/{self.node_id}")
                # for instance, signal in enumerate(signals, start=1):
                #     print(f"  Instance {instance}")
                #     for node in signal.node_history:
                #         print(f"    {node.node_id}")

            new_signal = Signal.combine_signals(signals)
            new_signal.add_node(self)
            if len(new_signal.ids) < new_signal.total_ids:
                output_signals.append(new_signal)

        # group children by color hash
        children_by_color_hash: Dict[str, List[Node]] = {}
        for child in self.children:
            children_by_color_hash.setdefault(child.color_hash, []).append(child)
        
        # send signals to children
        child_signals = {}
        for child_color_hash, children in children_by_color_hash.items():
            if len(children) > 1:
                signal_class = uuid.uuid4().hex # f"{self.node_id}/{child_color_hash[:8]}"
                for i, child in enumerate(children):
                    new_signal = Signal(
                        signal_class=signal_class,
                        ids=[i],
                        total_ids=len(children)
                    )
                    child_signals[child] = [*output_signals, new_signal]
            else:
                child_signals[children[0]] = output_signals
        
        return child_signals, patterns


def dag_to_nodes(graph: nx.DiGraph) -> List[Node]:
    """Convert a DAG to a list of nodes.

    Args:
        graph (nx.DiGraph): The DAG

    Returns:
        List[Node]: The nodes
    """
    # traverse in reverse topological order
    nodes: Dict[str, Node] = {}
    graph = graph.copy()
    # add src and sink nodes
    graph.add_node('src', color_hash='src')
    graph.add_node('sink', color_hash='sink')

    for node in graph.nodes:
        if node != 'src' and node != 'sink' and graph.in_degree(node) == 0:
            graph.add_edge('src', node)
        if node != 'src' and node != 'sink' and graph.out_degree(node) == 0:
            graph.add_edge(node, 'sink')

    top_sort = list(nx.topological_sort(graph))
    for node_id in top_sort[::-1]:
        node = graph.nodes[node_id]
        children = [nodes[child_id] for child_id in graph.successors(node_id)]
        nodes[node_id] = Node(
            node_id=node_id,
            color_hash=node['color_hash'],
            children=children
        )
    
    return [nodes[node_id] for node_id in top_sort]

def test_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node('A', color='A')
    graph.add_node('B1', color='B')
    graph.add_node('B2', color='B')
    graph.add_node('C1', color='C')
    graph.add_node('C2', color='C')
    graph.add_node('D1', color='D')
    graph.add_node('D2', color='D')
    graph.add_node('E', color='E')
    graph.add_node('F', color='F')

    graph.add_edge('A', 'B1')
    graph.add_edge('A', 'B2')
    graph.add_edge('B1', 'C1')
    graph.add_edge('B1', 'D1')
    graph.add_edge('B2', 'C2')
    graph.add_edge('B2', 'D2')
    graph.add_edge('C1', 'E')
    graph.add_edge('C2', 'E')
    graph.add_edge('D1', 'F')
    graph.add_edge('D2', 'F')

    return graph

thisdir = pathlib.Path(__file__).parent.resolve()
def main():
    # graph = load_dag_from_json(thisdir.joinpath('pegasus-instances', 'montage', 'chameleon-cloud', 'montage-chameleon-2mass-01d-001.json'))
    graph = test_graph()
    graph = set_color_hashes(graph)
    
    partial_patterns: Dict[str, List[Signal]] = {}
    signals: Dict[str, List[Signal]] = {}
    for node in dag_to_nodes(graph):
        new_signals, new_patterns = node.process_signals(signals.get(node, []))
        for child, child_signals in new_signals.items():
            signals.setdefault(child, []).extend(child_signals)
        for signal_class, pattern_signals in new_patterns.items():
            partial_patterns.setdefault(signal_class, []).extend(pattern_signals)

    # Merge partial patterns with overlapping node_history
    patterns: Dict[str, List[Signal]] = {}
    for signal_class, partial_pattern_signals in partial_patterns.items():
        all_node_ids = set().union(node.node_id for signal in partial_pattern_signals for node in signal.node_history)
        pattern_instances = list(nx.weakly_connected_components(graph.subgraph(all_node_ids)))
        pattern_hash = sha256(str(sorted(set(pattern_instances[0]))).encode()).hexdigest()
        patterns.setdefault(pattern_hash, []).extend(pattern_instances)

    # print patterns
    for pattern_hash, pattern_instances in patterns.items():
        print(f"Pattern {pattern_hash}")
        for instance, instance_nodes in enumerate(pattern_instances, start=1):
            print(f"  Instance {instance}")
            for node in instance_nodes:
                print(f"    {node}")

    print(f"found {len(patterns)} patterns")

    fig, ax = draw_workflow(graph, color_attr="color")
    plt.legend()
    # save figure to disk
    fig.savefig(thisdir.joinpath('workflow.png'))
    plt.close(fig)
    

if __name__ == "__main__":
    main()

