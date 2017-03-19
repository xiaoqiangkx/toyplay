# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: decision_tree_test.py
@time: 2017/3/18 16:22
@change_time:
1.2017/3/18 16:22
"""
from algorithm.DecisionTree import DecisionTree
from algorithm.DecisionTree import LeafNode
from algorithm.DecisionTree import DecideNode
from collections import defaultdict
import random
from utils import formula


def make_dataset(num, num_feature):
    """
    1. make a dataset with the amount of num
    and the amount of num_feature features
    2. every sample is a different category
    """
    data = []
    for i in xrange(num):
        temp = [random.randint(1, 5) for j in xrange(num_feature)]
        data.append(temp)

    return data


def make_tree_recursive(data, data_index_list, index_left_list, root_node, choose_func):
    if len(data_index_list) == 1 or len(index_left_list) == 0:
        leaf = LeafNode(root_node)
        leaf.num = len(data_index_list)
        root_node.add_node(leaf, None)
        return

    index = choose_func(index_left_list)
    new_index_left_list = [x for x in index_left_list]
    new_index_left_list.remove(index)
    root_node.decide_index = index

    types_dict = defaultdict(list)
    for idx in data_index_list:
        types_dict[data[idx][index]].append(idx)

    for key, index_list in types_dict.iteritems():
        if len(index_list) == 1 or len(new_index_left_list) == 0:   # 当然还有其他终止条件
            leaf = LeafNode(root_node)
            leaf.num = len(index_list)
            root_node.add_node(leaf, formula.equal(key))
            continue

        decide_node = DecideNode(root_node)
        root_node.add_node(decide_node, formula.equal(key))
        make_tree_recursive(data, index_list, new_index_left_list, decide_node, choose_func)
    return


def make_tree(data):
    """
    rule: choose the first remaining features left
    """
    tree = DecisionTree()
    index_left_list = range(len(data[0]))
    data_index_list = range(len(data))
    choose_func = lambda x: random.sample(x, 1)[0]
    make_tree_recursive(data, data_index_list, index_left_list, tree.root, choose_func=choose_func)
    return tree


if __name__ == '__main__':
    # Make a simple Decision Tree and plot it
    num = 10
    num_feature = 5
    data = make_dataset(num, num_feature)
    decision_tree = make_tree(data)
    decision_tree.show()
    dot_tree = decision_tree.save("test.dot")
    # print dot_tree.source
