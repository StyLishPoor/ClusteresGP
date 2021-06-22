import networkx as nx

class PGraph:
  PG_NUM = 0
  def __init__(self, Graph):
    self.id = PG_NUM++ 
    self.Graph = Graph
    self.bridge_nodes = []
    self.neighbors_bridge_nodes = {}
    self.source_ppr_vector = {}
    self.bridge_ppr_vector = []
    self.true_node_degree = {}

  
  def reg_bridge_nodes(self, bridge_nodes):
    self.bridges_nodes = bridge_nodes
    
  def reg_neighbors_bridge_nodes(self, neighbors_bridge_nodes):
    self.neighbors_bridge_nodes = neighbors_bridge_nodes
     
  def reg_true_node_degree(self, nodes, ALL_Graph):
    for node in nodes:
      self.true_node_degree[node] = ALL_Graph.degree[node] 

class Node:

