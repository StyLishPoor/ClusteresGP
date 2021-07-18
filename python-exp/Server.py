import Node
import random

class Server():
  server_num = 0
  
  def __init__(Nodes):
    self.Nodes = Nodes
    self.id = Server.server_num
    Server.server_num += 1

  def isLocal(self, node):
    return node.getServerId() == self.id

  def randamWalk(self, a, N, start):
    in_count = {}
    out_count = {}
    for node in self.Nodes:
      visit_count[node.getNodeId()] = 0
      #if node in start_bridge_nodes:
      #  out_graph_count[node] = {}

    RW_count = 0
    current_node = start
    next_node = start
    while (RW_count < N):
      current_node = next_node
      current_id = current_node.getNodeId()
      if (random.random() < a):
        in_count[current_id] += 1
        next_node= start
        RW_count += 1
      else:
        next_node= random.choice(current_node.getNeighbors())
        next_id = next_node.getNodeId()
        if (!self.isLocal(next_id)):
        #if (next_node not in start_G_nodes):
          #if (next_node.getNodeId() not in out_count[current_node]):
          if (next_id not in out_count[current_id]):
            out_count[current_id][next_id] = 1
          else:
            out_count[current_id][next_id] += 1
          next_node = start
          RW_count += 1

    return in_count, out_count
