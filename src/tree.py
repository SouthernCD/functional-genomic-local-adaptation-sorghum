import numpy as np
import copy
from Bio import Phylo
import matplotlib.pyplot as plt
from matplotlib import rcParams
import math
from collections import OrderedDict

rcParams['font.family'] = 'Arial'

rcParams['mathtext.it'] = 'Arial'
rcParams['mathtext.rm'] = 'Arial'
rcParams['mathtext.tt'] = 'Arial'
rcParams['mathtext.bf'] = 'Arial'
rcParams['mathtext.cal'] = 'Arial'
rcParams['mathtext.sf'] = 'Arial'
rcParams['mathtext.fontset'] = 'custom'

DEFAULT_STYLE_PARAMS = {
    # clade lines
    'bg_color': 'gray',
    'bg_lw': 0.5,
    'clade_colors': {},
    'clade_labels': {},
    'clade_lw': {},
    # tip labels
    'tip_label': False,
    'tip_fontsize': 6,
    'tip_color': {},
    'tip_bg_color': 'black',
    'tip_ha': 'right',
    'tip_va': 'center',
    'tip_name_map_dict': None,
    # tree
    'rotate_tree': 0,
    'use_edge_length': True,
}


def base_translate(num, base_num):
    """
    将一个十进制数num转换为base_num进制数
    base_translate(21,2)
    [1, 0, 1, 0, 1]
    """
    if num < base_num:
        return [num]
    else:
        n = int(math.log10(num) / math.log10(base_num)) + 1
        output_list = []
        for i in range(n):
            num_add = num // (base_num ** (n - i - 1))
            output_list.append(num_add)
            num = num - num_add * (base_num ** (n - i - 1))
        return output_list


def num_to_chr(num, digit):
    """
    when digit is 5:
    let 0 -> AAAAA
        1 -> AAAAB
    """

    base_list = base_translate(num, 26)

    output_list = []
    for i in [0]*(digit-len(base_list)) + base_list:
        output_list.append(chr(65+i))

    return ''.join(output_list)


def add_clade_name(tree, chr_name=False):
    used_name = []
    for clade in tree.find_clades():
        if clade.name:
            used_name.append(clade.name)

    num = 0
    for clade in tree.find_clades():
        if clade.name is None:
            if chr_name:
                new_id = "Node_%s" % num_to_chr(num, 3)
            else:
                new_id = "N%d" % num
            while new_id in used_name:
                num = num + 1
                if chr_name:
                    new_id = "Node_%s" % num_to_chr(num, 3)
                else:
                    new_id = "N%d" % num
            clade.name = new_id
            used_name.append(new_id)
    return tree


def lookup_by_names(tree, order='preorder'):
    """
    order ({'preorder', 'postorder', 'level'}) - Tree traversal order: 'preorder' (default) is depth-first search, 'postorder' is DFS with child nodes preceding parents, and 'level' is breadth-first search.

    order: depth-first search (DFS), preorder from root to first leaf one clade by clade, and second leaf... ; postorder, same search but from leaf to root
    level: breadth-frist search (BFS), from root to leaf one level by level

    see: https://biopython.org/DIST/docs/api/Bio.Phylo.BaseTree.TreeMixin-class.html
    """
    names = OrderedDict()
    for clade in tree.find_clades(order=order):
        if clade.name:
            if clade.name in names:
                raise ValueError("Duplicate key: %s" % clade.name)
            names[clade.name] = clade
    return names


def mid_unrooted_tree(tree):
    """
    root a unrooted tree at midpoint
    """
    tree.root_at_midpoint()
    tree_new = add_clade_name(tree)
    node_dict = lookup_by_names(tree_new)
    return tree_new, node_dict


def count_leaves(clade):
    """Recursively count the number of leaves in a clade."""
    if not clade.clades:
        return 1
    return sum(count_leaves(child) for child in clade.clades)


def plot_clade(clade, style_params, ax, xx, yy):
    bg_color = style_params.get('bg_color', '#FFFFFF')
    bg_lw = style_params.get('bg_lw', 0.5)
    clade_colors = style_params.get('clade_colors', {})

    """Plot each clade and its connections recursively."""
    for child in clade.clades:
        color = clade_colors.get(child.name, bg_color)
        ax.plot([xx[clade], xx[child]], [
                yy[clade], yy[child]], color=color, lw=bg_lw)
        plot_clade(child, style_params, ax, xx, yy)


def calculate_coordinate(clade, angle, axis, xx, yy, node_angles, node_axes, use_edge_length):
    """Recursive function to calculate (x, y) coordinates for each clade."""
    children = clade.clades
    if not children:
        return

    num_leaves = count_leaves(clade)
    start = axis - angle / 2
    for i, child in enumerate(children):
        h = child.branch_length if (
            child.branch_length and use_edge_length) else 1  # branch length
        child.parent = clade
        alpha = angle * count_leaves(child) / num_leaves
        beta = start + alpha / 2
        start += alpha

        xx[child] = h * np.cos(beta) + xx[clade]
        yy[child] = h * np.sin(beta) + yy[clade]

        node_angles[child] = alpha
        node_axes[child] = beta

        # Recursively calculate coordinates for child clade
        calculate_coordinate(child, alpha, beta, xx, yy,
                             node_angles, node_axes, use_edge_length)


def calculate_coordinates(tree, rotate_tree=0, use_edge_length=True):
    xx = {tree.root: 0}
    yy = {tree.root: 0}
    node_angles = {}
    node_axes = {}
    calculate_coordinate(tree.root, 2 * np.pi, rotate_tree,
                         xx, yy, node_angles, node_axes, use_edge_length)
    return xx, yy, node_angles, node_axes


class UnrootedTreePlotter:
    def __init__(self, tree, **kwargs):
        self.tree = copy.deepcopy(tree)

        # Set default style parameters
        self.style_params = DEFAULT_STYLE_PARAMS

        # Update style parameters
        self.style_params.update(kwargs)

    def plot(self, ax=None, save_file=None, **kwargs):
        # Set style parameters
        style_params = copy.deepcopy(self.style_params)
        style_params.update(kwargs)

        if not hasattr(self, 'xx'):
            self.mid_unrooted_tree()
            self.calculate_coordinates(style_params['rotate_tree'],
                                       style_params['use_edge_length'])

        # Plot the tree
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_aspect('equal')
            ax.axis('off')

        plot_clade(self.tree.root, style_params, ax, self.xx, self.yy)
        tip_label = style_params.get('tip_label', True)

        # Annotate the leaf nodes
        if tip_label == True:
            for clade in self.xx:
                if not clade.clades:  # Only label leaf nodes
                    tip_color = style_params['tip_color'].get(
                        clade.name, style_params['bg_color'])
                    # Calculate the angle for the text rotation
                    angle = self.node_axes[clade]
                    # Adjust the angle to make the text upright
                    if angle > np.pi/2:
                        angle -= np.pi
                    elif angle < -np.pi/2:
                        angle += np.pi
                    else:
                        angle = angle

                    plot_clade_name = style_params['tip_name_map_dict'][
                        clade.name] if style_params['tip_name_map_dict'] else clade.name

                    # if angle + np.pi/2 < 0:
                    if self.xx[clade] < self.xx[clade.parent]:
                        ax.text(self.xx[clade], self.yy[clade], plot_clade_name+" "*5, ha='right',
                                va='center', fontsize=style_params['tip_fontsize'], color=tip_color,
                                rotation=angle*180/np.pi, rotation_mode='anchor')
                    else:
                        ax.text(self.xx[clade], self.yy[clade], " "*5+plot_clade_name, ha='left',
                                va='center', fontsize=style_params['tip_fontsize'], color=tip_color,
                                rotation=angle*180/np.pi, rotation_mode='anchor')
                    # ax.text(self.xx[clade], self.yy[clade], clade.name, ha=style_params['tip_ha'],
                    #         va=style_params['tip_va'], fontsize=style_params['tip_fontsize'], color=tip_color,
                    #         rotation=angle*180/np.pi, rotation_mode='anchor')

        if save_file:
            fig.savefig(save_file, format='pdf', facecolor='none',
                        edgecolor='none', bbox_inches='tight')
        elif ax is None:
            plt.show()

        fig = ax.figure if ax is not None else fig
        return fig, ax

    def mid_unrooted_tree(self):
        self.tree, self.node_dict = mid_unrooted_tree(self.tree)

    def calculate_coordinates(self, rotate_tree=0, use_edge_length=True):
        self.xx, self.yy, self.node_angles, self.node_axes = calculate_coordinates(
            self.tree, rotate_tree, use_edge_length)
