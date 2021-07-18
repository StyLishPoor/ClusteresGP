import sys
import networkx as nx
import nxmetis
import Node
import Server

def main():
  G = nx.karate_club_graph()
  _, subgraphs = nxmetis.partition(G, int(sys.argv[1]))
  
  for num, subgraph in enumerate(subgraphs)::wq

    neighors = []
    for v in subgraph:
      neighors.append(
      node = Node.Node(num, v, list(G.neighbors(v))
    server = Server.Server(


if __name == '__main__':
  main()
