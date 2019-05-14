# -*- coding: utf-8 -*-
"""
@author: misiak

This script defines functions needed to create and display a symbolic
scheme of the electro-thermal system.
"""

import pydot
from PIL import Image

from .core_classes import System, Bath, Link

def sys_scheme(fp='scheme.png', display=True):
    """ Create and display the symbolic scheme of the System.

    Parameters
    ==========
    fp : str
        Filepath.

    See also
    ========
    Using pydot to create the scheme.
    Using Image to display the scheme.
    """
    graph = pydot.Dot(graph_name='test-detector',
                      rankdir = "LR",
                      graph_type='digraph')

    graph.set_node_defaults(shape='rectangle')
    graph.set_edge_defaults(arrowhead='none')

    bath_list = [e for e in System.elements_list if isinstance(e, Bath)]
    link_list = [e for e in System.elements_list if isinstance(e, Link)]

    node_dict = dict()
    for e in bath_list:
        lab = type(e).__name__ + ' ' + e.label
        node_dict[e] = pydot.Node(lab)
        graph.add_node(node_dict[e])

    for l in link_list:
        edge_l = pydot.Edge(node_dict[l.from_bath], node_dict[l.to_bath])
        lab = type(l).__name__ + ' ' + l.label
        edge_l.set_label(lab)
        graph.add_edge(edge_l)

    graph.write_png(fp)

    if display:
        try:
            image = Image.open(fp)
            image.show()
        except IOError as err:
            print('Cannot display scheme. IOError: {}'.format(err))
