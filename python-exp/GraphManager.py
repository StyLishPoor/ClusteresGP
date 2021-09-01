import sys
import random
import math
import networkx as nx

class GraphManager:
  GM_NUM = 0
  
  def __init__(self, Graph, out_nodes, bridge_nodes):
    self.id = GM_NUM++
    self.Graph = Graph 
    self.out_nodes = out_nodes 
    self.bridge_nodes = bridge_nodes 
  
  # 管理する頂点を始点とするRW
  def source_random_walk(self, source, N, a):
    in_visit_count = {}
    out_visit_count = {}
    for node in list(self.Graph.nodes):
      in_visit_count[node] = 0
      if node in self.bridge_nodes:
        out_visit_count[node] = {}

    RW_count = 0
    current_node = source
    next_node = source
    while (RW_count < N):
      current_node = next_node
      if (random.random() < a):
        in_visit_count[current_node] += 1
        next_node= source
        RW_count += 1
      else:
        next_node= random.choice(list(self.Graph.neighbors(current_node)))
        if (next_node in self.out_nodes):
          if (next_node not in out_visit_count[current_node]):
            out_visit_count[current_node][next_node] = 1
          else:
            out_visit_count[current_node][next_node] += 1
          next_node = source
          RW_count += 1
    return in_visit_count, out_visit_count

  # 管理する頂点が終点となるRW
  def bridge_random_walks(self, N, a):
    visit_counts = {} # visit_counts[N][M] : Bridge N → M の回数
    out_probability = {} # out_probability[N][M] : N → M にでていく確率

    for bridge in self.bridge_nodes:
      out_probability[bridge] = {}
      visit_count = {}
      out_graph_count = {}
      
      for node in self.Graph:
        visit_count[node] = 0

      for node in self.out_nodes:
        out_graph_count[node] = 0

      RW_count = 0
      out_count = 0
      current_node = bridge
      next_node = bridge
      while (RW_count < N):
        current_node = next_node
        if (random.random() < a):
          visit_count[current_node] += 1
          next_node= bridge
          RW_count += 1
        else:
          next_node= random.choice(list(self.Graph.neighbors(current_node)))
          if (next_node in self.out_nodes):
            out_graph_count[next_node] += 1
            out_count += 1
            next_node = bridge
            RW_count += 1

      for target, count in visit_count.items():
        visit_count[target] = count * N / (N - out_count)
      
      for node, count in out_graph_count.items():
        out_probability[bridge][node] = count / N
      
      visit_counts[bridge] = visit_count

    return visit_counts, out_probability

  def is_in_nodes(self, node):
    if node in self.in_nodes:
      return True
    return False
