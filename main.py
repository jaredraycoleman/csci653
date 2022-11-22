from copy import deepcopy
from itertools import combinations
import itertools
import json
import pathlib
from functools import lru_cache
from hashlib import sha256
import time
from typing import Dict, FrozenSet, Generator, Hashable, List, Optional, Set, Tuple, TypeVar, Union
import uuid

import networkx as nx
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
import pandas as pd
import plotly.express as px

from wfcommons.wfchef.find_microstructures import find_microstructures


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
    def __init__(self, src_node: "Node", child_color_hash: str, ids: List[Hashable], node_history: Set["Node"] = set()) -> None:
        self.src_node = src_node
        self.child_color_hash = child_color_hash
        self.ids = ids
        self.node_history = node_history
        self.dst_node = None

    @property
    def signal_class(self) -> str:
        return sha256(str((self.src_node.node_id, self.child_color_hash)).encode()).hexdigest()

    def add_node(self, node: "Node") -> None:
        self.node_history.add(node)

    def __repr__(self) -> str:
        return f"Signal {self.signal_class[:8]} (ids={self.ids})" #\nNodes: {list(self.node_history)}"
    
    @classmethod
    def combine_signals(self, signals: List["Signal"]) -> "Signal":
        assert len(set(signal.signal_class for signal in signals)) == 1
        return Signal(
            src_node=signals[0].src_node,
            child_color_hash=signals[0].child_color_hash,
            ids={id for signal in signals for id in signal.ids},
            node_history={node for signal in signals for node in signal.node_history}
        )

    def finalize(self, node: "Node") -> "Signal":
        self.dst_node = node
        return self

class Node:
    def __init__(self, node_id: str, color_hash: str, children: List["Node"]) -> None:
        self.node_id = node_id
        self.color_hash = color_hash
        self.children = children

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __repr__(self) -> str:
        return f"Node {self.node_id} (color_hash={self.color_hash[:8]})"

    def process_signals(self, signals: List[Signal]) -> Tuple[Dict[str, List[Signal]],List[Signal]]:
        """Process signals at this node.

        Args:
            signals (List[Signal]): The signals to process

        Returns:
            Dict[str, List[Signal]]: output signals. The key is child node id the value is the list of signals.
        """
        # group signals by signal class, then ids
        signals_by_class_and_id: Dict[str, Dict[Tuple[int], List[Signal]]] = {}
        for signal in signals: # Runtime: O(n)
            signals_by_class_and_id.setdefault(signal.signal_class, {}).setdefault(tuple(sorted(signal.ids)), []).append(signal) # Runtime: O(1)

        # Merge signals with the same ids. Runtime: O(n)
        signals_by_class = {
            signal_class: [
                Signal.combine_signals(id_signals)
                for id_signals in signals_by_id.values()
            ]
            for signal_class, signals_by_id in signals_by_class_and_id.items()
        }

        # process signals by signal class
        partial_pattern_instances = []
        output_signals: List[Signal] = []
        # Runtime: O(n) - each signal is included in one combination
        for signal_class, signals in signals_by_class.items():
            if len(signals) > 1:
                partial_pattern_instances.extend([signal.finalize(self) for signal in signals])

            new_signal = Signal.combine_signals(signals)
            new_signal.add_node(self)
            output_signals.append(new_signal)

        # group children by color hash
        # Runtime: O(m) - m is the number of children
        children_by_color_hash: Dict[str, List[Node]] = {}
        for child in self.children:
            children_by_color_hash.setdefault(child.color_hash, []).append(child)
        
        # send signals to children
        child_signals = {}
        # Runtime: O(n*m) - send each signal (maybe +1) to each child
        for child_color_hash, children in children_by_color_hash.items():
            if len(children) > 1:
                for i, child in enumerate(children):
                    new_signal = Signal(
                        src_node=self,
                        child_color_hash=child_color_hash,
                        ids=[uuid.uuid4().hex]
                    )
                    child_signals[child] = [*output_signals, new_signal]
            else:
                child_signals[children[0]] = output_signals
        
        return child_signals, partial_pattern_instances


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

T = TypeVar('T')
def merge_sets(sets: Set[Tuple[FrozenSet[T], FrozenSet[T], FrozenSet[T]]]) -> Set[FrozenSet[T]]:
    """
    Merges sets of sets.

    Args:
        sets (Set[FrozenSet[T]]): sets of sets.

    Returns:
        Set[FrozenSet[T]]: merged sets.
    """
    subsets, supersets = set(), set()
    for s, src, dst in sets:
        if any(s|src|dst < s2|src2|dst2 for s2, src2, dst2 in sets):
            print("Is subset: ", s, src, dst)
            subsets.add((s, src, dst))
        else:
            supersets.add((s, src, dst))
    
    merged = set()
    for s, src, dst in supersets:
        union = {*s}
        union_src = {*src}
        union_dst = {*dst}
        for s2, src2, dst2 in list(merged):
            if s2 & s:
                merged.remove((s2, src2, dst2))
                union |= s2
                union_src |= src2
                union_dst |= dst2
            
        merged.add((frozenset(union), frozenset(union_src), frozenset(union_dst)))

    if subsets:
        merged.update(merge_sets(subsets))
    return merged

def find_patterns(graph: nx.DiGraph) -> Dict[str, List[Signal]]:
    graph = set_color_hashes(graph)

    partial_pattern_instances: List[Signal] = []
    signals: Dict[str, List[Signal]] = {}
    for node in dag_to_nodes(graph):
        new_signals, new_partial_pattern_instances = node.process_signals(signals.get(node, []))
        for child, child_signals in new_signals.items():
            signals.setdefault(child, []).extend(child_signals)
        partial_pattern_instances.extend(new_partial_pattern_instances)

    partial_pattern_instances_node_ids = set()
    for signal in partial_pattern_instances:
        if signal.dst_node is None:
            raise ValueError('Signal has no destination node')
        partial_pattern_instances_node_ids.add(
            (
                frozenset({node.node_id for node in signal.node_history}),
                frozenset({signal.src_node.node_id}),
                frozenset({signal.dst_node.node_id})
            )
        )

    print(f"Found {len(partial_pattern_instances_node_ids)} partial patterns")
    # pattern_instances = merge_partial_patterns(partial_pattern_instances_node_ids)
    pattern_instances = merge_sets(partial_pattern_instances_node_ids)

    # remove any pattern instances which have edges to non-internal nodes
    for pattern_instance, src_nodes, dst_nodes in deepcopy(pattern_instances):
        all_nodes = {*pattern_instance, *src_nodes, *dst_nodes}
        all_neighbors = {neighbor for node in pattern_instance for neighbor in graph.neighbors(node)}
        if not all_neighbors.issubset(all_nodes):
            pattern_instances.remove((pattern_instance, src_nodes, dst_nodes))

    # pattern_instances = list(merge_partial_patterns_2(
    #     graph, 
    #     {
    #         frozenset({node.node_id for node in signal.node_history})
    #         for signal in partial_pattern_instances
    #     }
    # ))

    patterns: Dict[str, Set[FrozenSet[str]]] = {}
    for pattern_instance, *_ in pattern_instances:
        color_hashes = {graph.nodes[node_id]['color_hash'] for node_id in pattern_instance}
        pattern_hash = sha256(str(sorted(color_hashes)).encode()).hexdigest()        
        patterns.setdefault(pattern_hash, set()).add(pattern_instance)

    return patterns

def find_patterns_new(graph: nx.DiGraph) -> Dict[str, List[Set[str]]]:
    nodes_by_type_hash: Dict[str, Set[str]] = {}
    for node in graph.nodes:
        # th = graph.nodes[node]["type_hash"]
        # nodes_by_type_hash.setdefault(th, set()).add(node)
        for child in graph.successors(node):
            th = graph.nodes[child]["type_hash"]
            nodes_by_type_hash.setdefault((node, th), set())
            nodes_by_type_hash[(node, th)].add(child)

    for (node, th), children in nodes_by_type_hash.items():
        print(node, th, children)

def find_patterns_old(graph: nx.DiGraph) -> Dict[str, List[Signal]]:    
    # Add src and sink nodes
    graph.add_node('src', color='src')
    graph.add_node('sink', color='sink')
    for node in graph.nodes:
        if node not in ["src", "sink"] and graph.in_degree(node) == 0:
            graph.add_edge('src', node, color="src")
        if node not in ["src", "sink"] and graph.out_degree(node) == 0:
            graph.add_edge(node, 'sink', color="sink")

    graph = set_color_hashes(graph)
    nx.set_node_attributes(graph, {node: graph.nodes[node]['color_hash'] for node in graph.nodes}, 'type_hash')

    return find_microstructures(graph, verbose=False)

def print_patterns(patterns: Dict[str, List[Set[str]]]) -> None:
    num_pattern_instances = 0
    for pattern_hash, pattern_instances in patterns.items():
        print(f"Pattern {pattern_hash}")
        for instance, instance_nodes in enumerate(pattern_instances, start=1):
            num_pattern_instances += 1
            print(f"  Instance {instance}")
            for node in instance_nodes:
                print(f"    {node}")
            print("  ...")
            break

thisdir = pathlib.Path(__file__).parent.resolve()
def main():
    graph = load_dag_from_json(thisdir.joinpath('pegasus-instances', 'montage', 'chameleon-cloud', 'montage-chameleon-dss-125d-001.json'))
    # graph = test_graph()
    pegasus_path = thisdir.joinpath('pegasus-instances')
    app_paths = [path for path in pegasus_path.iterdir() if path.is_dir()]

    records = []
    for app_path in app_paths:
        chameleon_path = app_path.joinpath("chameleon-cloud")
        if not chameleon_path.exists():
            continue

        if app_path.name != "montage":
            continue

        print("Processing Application:", app_path.name)
        graphs = [load_dag_from_json(path) for path in chameleon_path.glob('*.json')]
        graphs = sorted(graphs, key=lambda graph: graph.size())
        for graph in graphs:
            t = time.time()
            patterns = find_patterns(graph.copy())
            t = time.time() - t
            
            records.append({
                # 'graph': graph_file.name,
                "app": app_path.name,
                'num_nodes': len(graph.nodes),
                'num_edges': len(graph.edges),
                'num_patterns': len(patterns),
                'num_pattern_instances': sum(len(pattern) for pattern in patterns.values()), 
                'time': t, 
                'method': 'new'
            })

            print(f"NEW: Found {len(patterns)} patterns and {sum(len(pattern) for pattern in patterns.values())} pattern instances in {t:.2f} seconds")
            print_patterns(patterns)
            
            t = time.time()
            patterns = find_patterns_old(graph.copy())
            t = time.time() - t

            records.append({
                # 'graph': graph_file.name,
                "app": app_path.name,
                'num_nodes': len(graph.nodes),
                'num_edges': len(graph.edges),
                'num_patterns': len(patterns),
                'num_pattern_instances': sum(len(pattern) for pattern in patterns.values()), 
                'time': t, 
                'method': 'old'
            })

            print(f"\nOLD: Found {len(patterns)} patterns and {sum(len(pattern) for pattern in patterns.values())} pattern instances in {t:.2f} seconds")
            print_patterns(patterns)

            # fig, ax = draw_workflow(graph, color_attr="color")
            # plt.legend()
            # fig.savefig(f"{app_path.name}-{graph.size()}.png")
            # plt.close(fig)

            break


    return
    df = pd.DataFrame.from_records(records).sort_values(['app', 'num_nodes'])
    df.to_csv('pegasus-patterns.csv', index=False)
    print(df)

    fig = px.line(
        df, x='num_nodes', y='time', line_dash='method',
        facet_col='app', facet_col_wrap=3, facet_row_spacing=0.1, facet_col_spacing=0.1,
        labels={'num_nodes': 'Number of Nodes', 'time': 'Time (s)'},
        title='Time to Find Patterns',
        template='plotly_white'
    )
    # set line color to black
    fig.for_each_trace(lambda trace: trace.update(line_color='black'))
    fig.write_image('pegasus-patterns.png')
    fig.write_html('pegasus-patterns.html')
    

if __name__ == "__main__":
    main()

