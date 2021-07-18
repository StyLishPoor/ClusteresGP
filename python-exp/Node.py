class Node():
  def __init__(self, server_id, node_id, neighbors):
    self.server = server_id;
    self.id = node_id
    self.neighbors = neighbors
  
  def getNodeId(self):
    return self.id

  def getServerId(self):
    return self.server

  def getNeighbors(self):
    return self.neighbors

  def getNodeDegree(self):
    return len(self.neighbors)
