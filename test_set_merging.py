from ast import In
from copy import deepcopy
import pprint
import random
from typing import TypeVar, Set, FrozenSet

T = TypeVar('T')
def merge_sets(sets: Set[FrozenSet[T]]) -> Set[FrozenSet[T]]:
    """
    Merges sets of sets.

    Args:
        sets (Set[FrozenSet[T]]): sets of sets.

    Returns:
        Set[FrozenSet[T]]: merged sets.
    """
    subsets, supersets = set(), set()
    for s in sets:
        if any(s < s2 for s2 in sets):
            subsets.add(s)
        else:
            supersets.add(s)
    
    merged = set()
    for s in supersets:
        overlapping = {s2 for s2 in merged if s2 & s}
        merged  -= overlapping
        merged.add(frozenset.union(s, *overlapping))

    if subsets:
        merged.update(merge_sets(subsets))
    return merged


def random_sets(n: int, m: int, maxval: int) -> Set[FrozenSet[int]]:
    """
    Creates a set of random sets.

    Args:
        n (int): number of sets.
        m (int): number of elements in each set.
        maxval (int): maximum value of an element.

    Returns:
        Set[FrozenSet[int]]: set of sets.
    """
    import random
    return {frozenset(random.sample(range(maxval), m)) for _ in range(n)}

def main():

    for i in range(10000):
        sets = list(random_sets(20, 3, 100))
        # random permutation of sets
        all_merged_sets = set()
        for _ in range(50):
            random.shuffle(sets)
            all_merged_sets.add(frozenset(merge_sets(sets)))

        if i < 1 or len(all_merged_sets) > 1:
            if len(all_merged_sets) > 1:
                print('Found counterexample!')
            print("original sets:")
            for s in sets:
                print(sorted(s))
            print("merged sets:")
            for merged_sets in all_merged_sets:
                for sets in sorted(merged_sets):
                    print(sorted(sets))
                print()
            
            if len(all_merged_sets) > 1:
                break


if __name__ == "__main__":
    main()