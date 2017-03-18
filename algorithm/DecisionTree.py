# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: DecisionTreeClassifier.py
@time: 2017/3/18 12:57
@change_time:
1.2017/3/18 12:57 Make a simple decision tree
"""
from utils import logger
from collections import deque
from graphviz import Digraph


class DecisionNode(object):
    """
    """

    def __init__(self, parent=None):
        self.parent = parent
        self._children = []
        self._decide_info = []
        return

    def is_leaf(self):
        return False

    def is_decide(self):
        return False

    def get_decide_func(self):
        if not self.parent:
            print "not parent"
            return None

        return self.parent.get_child_decide_func(self)

    def index(self, child):
        return self._children.index(child)

    @property
    def children(self):
        return self._children

    def add_node(self, node, func):
        self.children.append(node)
        self._decide_info.append(func)
        return


class DecideNode(DecisionNode):
    def __init__(self, parent, decide_index=None):
        super(DecideNode, self).__init__(parent=parent)
        self._decide_index = decide_index
        self._decide_info = []
        return

    # def decide(self, x):
    #     for index, func in self.children:
    #         feature = x[index]
    #         if func(feature):
    #             return self.children[index]
    #
    #     return None

    def is_decide(self):
        return True

    def get_child_decide_func(self, child):
        func_index = self.index(child)
        if func_index is None:
            return None
        return self._decide_info[func_index]

    def __repr__(self):
        if not self.parent:
            return "R index({0})".format(self._decide_index)

        return "D index({0})".format(self._decide_index)


class LeafNode(DecisionNode):

    def __init__(self, parent, category=None):
        super(LeafNode, self).__init__(parent)
        self._category = category
        self._num = 1
        return

    def is_leaf(self):
        return True

    @property
    def num(self):
        return

    @num.setter
    def num(self, value):
        self._num = value
        return

    def __repr__(self):
        return "L category({0}), num({1})".format(self._category, self._num)


class DecisionTree(object):

    def __init__(self):
        self._root = DecideNode(None, 0)
        self._depth = 0
        return

    @property
    def root(self):
        return self._root

    def get_node_queue_with_level(self):
        level = 1
        node_queue = deque()
        result_queue = deque()
        node_queue.append(self._root)
        node_queue.append(level)
        result_queue.append(self._root)
        result_queue.append(level)

        while len(node_queue):
            node = node_queue.popleft()
            level = node_queue.popleft()
            for child in node.children:
                node_queue.append(child)
                node_queue.append(level + 1)
                result_queue.append(child)
                result_queue.append(level + 1)

        return result_queue

    def get_node_queue(self):
        node_queue = deque()
        result_queue = deque()
        node_queue.append(self._root)
        result_queue.append(self._root)

        while len(node_queue):
            node = node_queue.popleft()
            for child in node.children:
                node_queue.append(child)
                result_queue.append(child)

        return result_queue

    def show(self):
        """use queue to traversal"""
        node_queue = self.get_node_queue_with_level()

        cur_level = 1
        cur_list = []
        while len(node_queue):
            node = node_queue.popleft()
            level = node_queue.popleft()

            if cur_level != level:
                print "\t".join(cur_list)
                cur_level = level
                cur_list = []

            cur_list.append(repr(node))

        if cur_list:
            print "\t".join(cur_list)
        return

    def save(self, filename):
        dot_tree = Digraph(comment="decision_tree_test")
        node_queue = self.get_node_queue()

        for item in node_queue:
            key_item = str(id(item))
            dot_tree.node(key_item, repr(item))
            for index, child in enumerate(item.children):
                key_child = str(id(child))
                decide_func = child.get_decide_func()
                dot_tree.edge(key_item, key_child, label=repr(decide_func))

        dot_tree.render(filename, view=True)
        return dot_tree
