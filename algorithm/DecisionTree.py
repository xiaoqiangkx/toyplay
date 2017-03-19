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
from collections import defaultdict
from utils import formula
import random
from collections import Counter


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
        return self.children.index(child)

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

    @property
    def decide_index(self):
        return self._decide_index

    @decide_index.setter
    def decide_index(self, value):
        self._decide_index = value
        return

    def is_decide(self):
        return True

    def get_child_decide_func(self, child):
        func_index = self.index(child)
        if func_index is None:
            return None
        return self._decide_info[func_index]

    def __repr__(self):
        if not self.parent:
            return "R index({0})".format(self.decide_index)

        return "D index({0})".format(self.decide_index)


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
        return self._num

    @property
    def category(self):
        return self._category

    @category.setter
    def category(self, value):
        self._category = value
        return

    @num.setter
    def num(self, value):
        self._num = value
        return

    def __repr__(self):
        return "L category({0}), num({1})".format(self.category, self.num)


def choose_random(data, target, data_index_list, index_left_list):
    sample = random.sample(index_left_list, 1)
    if not sample:
        logger.error("failed to choose sample for index_left_list({0})".format(index_left_list))
        return None
    return sample[0]

CHOOSE_RANDOM = 1
CHOOSE_FUNC_DICT = {
    CHOOSE_RANDOM: choose_random,
}


def get_choose_func(key):
    return CHOOSE_FUNC_DICT.get(key, None)


class DecisionTree(object):

    def __init__(self, depth=10):
        self._root = DecideNode(None, 0)
        self._max_depth = depth
        return

    @property
    def max_depth(self):
        return self._max_depth

    def make_tree(self, data, target, choose_func=CHOOSE_RANDOM):
        index_left_list = range(len(data[0]))
        data_index_list = range(len(data))
        depth = 1
        self.make_tree_recursive(data, target, data_index_list, index_left_list, self.root, depth, choose_func)
        return

    def choose_index(self, choose_func, data, target, data_index_list, index_left_list):
        choose_func = get_choose_func(choose_func)

        if not choose_func:
            logger.error("failed to find choose_func for key({0})".format(choose_func))
            return None

        index = choose_func(data, target, data_index_list, index_left_list)
        if index is None:
            logger.error("failed to find index with data_index_list({0}), index_left_list({1}), choose_func({2})"
                         .format(data_index_list, index_left_list, choose_func))
            return None
        return index

    def check_finish(self, data, target, data_index_list, index_left_list, depth):

        if len(data_index_list) <= 1:   # left one data
            return True

        cnt = Counter()
        for index in data_index_list:
            cnt[target[index]] += 1

        if len(cnt) == 1:               # left one category
            return True

        if len(index_left_list) == 0:   # no attribute left
            return True

        if depth == self.max_depth:     # reach the max depth
            return True

        return False

    def get_majority_category(self, target, data_index_list):
        cnt = Counter()
        for index in data_index_list:
            cnt[target[index]] += 1

        most_common_list = cnt.most_common(1)
        if not most_common_list:
            logger.error("can't not find most_common_list for data({0})".format(data_index_list))
            return None

        category = most_common_list[0][0]
        return category

    def make_tree_recursive(self, data, target, data_index_list, index_left_list, root_node, depth, choose_func):
        if self.check_finish(data, target, data_index_list, index_left_list, depth):
            leaf = LeafNode(root_node)
            category = self.get_majority_category(target, data_index_list)
            leaf.num = len(data_index_list)
            leaf.category = category
            root_node.add_node(leaf, None)
            return

        index = self.choose_index(choose_func, data, target, data_index_list, index_left_list)
        if index is None:
            return

        new_index_left_list = [x for x in index_left_list]
        new_index_left_list.remove(index)
        root_node.decide_index = index

        types_dict = defaultdict(list)
        for idx in data_index_list:
            types_dict[data[idx][index]].append(idx)

        for key, new_index_list in types_dict.iteritems():
            new_depth = depth + 1
            if self.check_finish(data, target, new_index_list, new_index_left_list, new_depth):  # 当然还有其他终止条件
                leaf = LeafNode(root_node)
                leaf.num = len(new_index_list)
                category = self.get_majority_category(target, new_index_list)
                leaf.category = category
                root_node.add_node(leaf, formula.equal(key))
                continue

            decide_node = DecideNode(root_node)
            root_node.add_node(decide_node, formula.equal(key))
            self.make_tree_recursive(data, target, new_index_list, new_index_left_list, decide_node, new_depth, choose_func)
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
