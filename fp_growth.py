from collections import defaultdict, namedtuple
import pygraphviz as pgv
import copy
import random
from PIL import Image
import cv2

def find_frequent_itemsets(transactions, minimum_support, include_support=False):
    items = defaultdict(lambda: 0)  # mapping from items to their supports

    # if using support rate instead of support count
    if 0 < minimum_support <= 1:
        minimum_support = minimum_support * len(transactions)

    for transaction in transactions:
        for item in transaction:
            items[item] += 1

    # Remove infrequent items from the item support dictionary.
    items = dict(
        (item, support) for item, support in items.items() if support >= minimum_support
    )

    # Build our FP-tree. Before any transactions can be added to the tree, they
    # sorted in decreasing order of frequency.
    def clean_transaction(transaction):
        transaction = filter(lambda v: v in items, transaction)
        transaction = sorted(transaction, key=lambda v: items[v], reverse=True)
        return transaction  

    print("Transaction: ")
    print(transaction)

    master = FPTree()
    for transaction in list(map(clean_transaction, transactions)):
        master.add(transaction)
        # visual_fptree(master,"foo")

    # Visual FPTree without route
    tree = copy.copy(master)
    visual_fptree(tree,"foo")

    solve = []
    for item,nodes in master.items():
        print(item+": " , end="")
        for path in master.prefix_paths(item):
            print( "[ ", end="")
            for i in path:
                print(i.item+" ", end="")
            print("] ", end="")
        print()

    def find_with_suffix(tree, suffix):
        for item, nodes in tree.items():
            support = sum(n.count for n in nodes)
            if support >= minimum_support and item not in suffix:
                # New winner!
                found_set = [item] + suffix
                yield (found_set, support) if include_support else found_set
                print("Frequent pattern: ", end="")
                print(found_set, support)
                
                # Build a conditional tree and recursively search for frequent
                # itemsets within it.

                tree_prefix_paths = []

                # Remove node-end path for ONLY visual
                for i in tree.prefix_paths(item):
                    tree_prefix_paths.append(i)

                c = 0
                for i in tree_prefix_paths:
                    if len(i)!=0:
                        c+=1

                if c != 0:
                    cond_tree = conditional_tree_from_paths(tree_prefix_paths)
                
                    print("Path: ", end="")
                    for path in tree_prefix_paths:
                        print("[ ", end="")
                        for i in path:
                            print(i.item + " ", end="")
                        print("] ", end="")
                    print()
                    # visual_fptree(cond_tree,str(item), item)
                    # visual_fptree(cond_tree,str(item))
                    
                    for s in find_with_suffix(cond_tree, found_set):
                        yield s  # pass along the good news to our caller

    # Search for frequent itemsets, and yield the results we find.
    for itemset in find_with_suffix(master, []):
        yield itemset


def visual_fptree(tree,file_name, item=None):
    A = pgv.AGraph(directed=True, strict=True, rankdir="LR")
    queue = []

    A.add_node("root");

    for i in tree.root.children:
        queue.append(i)
        A.add_edge("root", i.name)
        A.get_node(i.name).attr["label"] = str(i.item) + ":" + str(i.count)

    while queue:
        l, queue = queue[:], []
        for s in l:
            # print(s.item)
            A.get_node(s.name).attr["label"] = str(s.item) + ":" + str(s.count)
            for i in s.children:
                if item is not None:
                    if i.item != item:
                        queue.append(i)
                        A.add_edge(s.name, i.name)
                        A.get_node(i.name).attr["label"] = str(i.item) + ":" + str(i.count)
                else:
                    queue.append(i)
                    A.add_edge(s.name, i.name)
                    A.get_node(i.name).attr["label"] = str(i.item) + ":" + str(i.count)

    A.graph_attr["epsilon"] = "0.001"
    # print(A.string())  # print dot file to standard output
    A.layout("dot")  # layout with dot
    A.draw(file_name+".png")  # write to file

    image = cv2.imread(file_name+".png")
    cv2.imshow(file_name, image)
    cv2.waitKey(0)


class FPTree(object):
    Route = namedtuple("Route", "head tail")

    def __init__(self):
        # The root node of the tree.
        self._root = FPNode(self, None, None)

        # A dictionary mapping items to the head and tail of a path of
        # "neighbors" that will hit every node containing that item.
        self._routes = {}
        self._count = 0
    @property
    def root(self):
        """The root node of the tree."""
        return self._root

    def add(self, transaction):
        """Add a transaction to the tree."""
        point = self._root
        # print("Transactions: ", transaction , end=" ")
        for item in transaction:
            # try:
            #     self._count+=1
            #     visual_fptree(self._root.tree, str(self._count))
            # except Exception as e:
            #     print(e)

            # print(item, " ", end="")

            next_point = point.search(item)
            if next_point:
                # There is already a node in this tree for the current
                # transaction item; reuse it.
                next_point.increment()
            else:
                # Create a new point and add it as a child of the point we're
                # currently looking at.
                next_point = FPNode(self, item)
                point.add(next_point)

                # Update the route of nodes that contain this item to include
                # our new node.
                self._update_route(next_point)

            point = next_point
        print()

    def _update_route(self, point):
        """Add the given node to the route through all nodes for its item."""
        assert self is point.tree

        try:
            route = self._routes[point.item]
            route[1].neighbor = point  # route[1] is the tail
            self._routes[point.item] = self.Route(route[0], point)
        except KeyError:
            # First node for this item; start a new route.
            self._routes[point.item] = self.Route(point, point)

    def items(self):
        for item in self._routes.keys():
            yield (item, self.nodes(item))

    def nodes(self, item):
        try:
            node = self._routes[item][0]
        except KeyError:
            return

        while node:
            yield node
            node = node.neighbor

    def prefix_paths(self, item):
        """Generate the prefix paths that end with the given item."""

        def collect_path(node):
            path = []
            while node and not node.root:
                path.append(node)
                node = node.parent
            path.reverse()
            return path

        return (collect_path(node) for node in self.nodes(item))

    def inspect(self):
        print("Tree:")
        self.root.inspect(1)

        print()
        print("Routes:")
        for item, nodes in self.items():
            print("  %r" % item)
            for node in nodes:
                print("    %r" % node)


def conditional_tree_from_paths(paths):
    """Build a conditional FP-tree from the given prefix paths."""
    tree = FPTree()
    condition_item = None
    items = set()

    # Import the nodes in the paths into the new tree. Only the counts of the
    # leaf notes matter; the remaining counts will be reconstructed from the
    # leaf counts.
    for path in paths:
        if condition_item is None:
            condition_item = path[-1].item

        point = tree.root
        for node in path:
            next_point = point.search(node.item)
            if not next_point:
                # Add a new node to the tree.
                items.add(node.item)
                count = node.count if node.item == condition_item else 0
                next_point = FPNode(tree, node.item, count)
                point.add(next_point)
                tree._update_route(next_point)
            point = next_point

    assert condition_item is not None

    # Calculate the counts of the non-leaf nodes.
    for path in tree.prefix_paths(condition_item):
        count = path[-1].count
        for node in reversed(path[:-1]):
            node._count += count

    return tree


class FPNode(object):
    def __init__(self, tree, item, count=1):
        self._tree = tree
        self._item = item
        self._count = count
        self._parent = None
        self._children = {}
        self._neighbor = None
        # For visual tree
        self._name = random.random()*10000

    def add(self, child):
        if not isinstance(child, FPNode):
            raise TypeError("Can only add other FPNodes as children")

        if child.item not in self._children:
            self._children[child.item] = child
            child.parent = self

    def search(self, item):
        try:
            return self._children[item]
        except KeyError:
            return None

    def __contains__(self, item):
        return item in self._children

    @property
    def tree(self):
        return self._tree

    @property
    def item(self):
        return self._item

    @property
    def count(self):
        return self._count

    @property
    def name(self):
        return self._name

    def increment(self):
        if self._count is None:
            raise ValueError("Root nodes have no associated count.")
        self._count += 1

    @property
    def root(self):
        return self._item is None and self._count is None

    @property
    def leaf(self):
        return len(self._children) == 0

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        if value is not None and not isinstance(value, FPNode):
            raise TypeError("A node must have an FPNode as a parent.")
        if value and value.tree is not self.tree:
            raise ValueError("Cannot have a parent from another tree.")
        self._parent = value

    @property
    def neighbor(self):
        return self._neighbor

    @neighbor.setter
    def neighbor(self, value):
        if value is not None and not isinstance(value, FPNode):
            raise TypeError("A node must have an FPNode as a neighbor.")
        if value and value.tree is not self.tree:
            raise ValueError("Cannot have a neighbor from another tree.")
        self._neighbor = value

    @property
    def children(self):
        return tuple(self._children.values())

    def inspect(self, depth=0):
        print(("  " * depth) + repr(self))
        for child in self.children:
            child.inspect(depth + 1)

    def __repr__(self):
        if self.root:
            return "<%s (root)>" % type(self).__name__
        return "<%s %r (%r)>" % (type(self).__name__, self.item, self.count)


def subs(l):
    assert type(l) is list
    if len(l) == 1:
        return [l]
    x = subs(l[1:])
    return x + [[l[0]] + y for y in x]


# Association rules
def assoc_rule(freq, min_conf=0.6):
    assert type(freq) is dict
    result = []
    for item, sup in freq.items():
        print("Sub list from " + str(item) +":", end="")
        print(subs(list(item)))
        for subitem in subs(list(item)):
            sb = [x for x in item if x not in subitem]
            if sb == [] or subitem == []:
                continue
            if len(subitem) == 1 and (subitem[0][0] == "in" or subitem[0][0] == "out"):
                continue
            conf = sup / freq[tuple(subitem)]
            if conf >= min_conf:
                result.append({"from": subitem, "to": sb, "sup": sup, "conf": conf})
    return result


if __name__ == "__main__":
    from optparse import OptionParser
    import csv

    p = OptionParser(usage="%prog data_file")
    p.add_option(
        "-s",
        "--minimum-support",
        dest="minsup",
        type="int",
        help="Minimum itemset support (default: 2)",
    )
    p.add_option(
        "-n",
        "--numeric",
        dest="numeric",
        action="store_true",
        help="Convert the values in datasets to numerals (default: false)",
    )
    p.add_option(
        "-c",
        "--minimum-confidence",
        dest="minconf",
        type="float",
        help="Minimum rule confidence (default 0.6)",
    )
    p.add_option(
        "-f",
        "--find",
        dest="find",
        type="str",
        help="Finding freq(frequency itemsets) or rule(association rules) (default: freq)",
    )
    p.set_defaults(minsup=2)
    p.set_defaults(numeric=False)
    p.set_defaults(minconf=0.6)
    p.set_defaults(find="freq")
    options, args = p.parse_args()

    assert options.find == "freq" or options.find == "rule"

    if len(args) < 1:
        p.error("must provide the path to a CSV file to read")

    transactions = []
    with open(args[0]) as database:
        for row in csv.reader(database):
            if options.numeric:
                transaction = []
                for item in row:
                    transaction.append(int(item))
                transactions.append(transaction)
            else:
                transactions.append(row)

    result = []
    res_for_rul = {}
    for itemset, support in find_frequent_itemsets(transactions, options.minsup, True):
        result.append((itemset, support))
        res_for_rul[tuple(itemset)] = support

    if options.find == "freq":
        result = sorted(result, key=lambda i: i[0])
        for itemset, support in result:
            print(str(itemset) + " " + str(support))
    if options.find == "rule":
        rules = assoc_rule(res_for_rul, options.minconf)
        for ru in rules:
            print(str(ru["from"]) + " -> " + str(ru["to"]))
            print("support = " + str(ru["sup"]) + "confidence = " + str(ru["conf"]))
