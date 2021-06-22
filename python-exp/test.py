import matplotlib.pyplot as plt
import graphviz
import networkx as nx
import nxmetis
import sys
import random

def source_GM_ppr(sub_graph, ppr_source, bridge_nodes):
  # ppr vector of ppr_source and bridge_nodes
  ppr_vector= {}
  ppr_vector[ppr_source] = nx.pagerank(sub_graph, personalization={ppr_source:1}) 

  for bridge_node in bridge_nodes:
    ppr_vector[bridge_node] = nx.pagerank(sub_graph, personalization={bridge_node:1})

  return ppr_vector

def bridge_GM_ppr(sub_graph, bridge_nodes):
  # ppr vector of bridge_nodes
  ppr_vector= {}

  for bridge_node in bridge_nodes:
    ppr_vector[bridge_node] = nx.pagerank(sub_graph, personalization={bridge_node:1})

  return ppr_vector
  
def merge(ppr_source, source_bridge_nodes, bridge_edges, source_ppr_inverse, source_ppr_vector, bridge_ppr_vectors, ALL_Graph):
  ppr_result = {}
  for target in source_ppr_vector[ppr_source]:
    ppr_result[target] = source_ppr_inverse[target] 
    for bridge in source_bridge_nodes:
      ppr_result[target] += source_ppr_inverse[bridge] * source_ppr_vector[bridge][target] 

  for bridge_ppr_vector in bridge_ppr_vectors:
    for bridge_node in bridge_ppr_vector:
      for bridge in source_bridge_nodes:
        if ((bridge_node, bridge) in bridge_edges or (bridge, bridge_node) in bridge_edges):
          for target in bridge_ppr_vector[bridge_node]:
            if (target not in ppr_result):
              ppr_result[target] = 0
            #ppr_result[target] += (1/G.degree(bridge)) * source_ppr_vector[ppr_source][bridge] * bridge_ppr_vector[bridge_node][target]
            ppr_result[target] += (1/G.degree(bridge)) * source_ppr_inverse[bridge] * bridge_ppr_vector[bridge_node][target]
            #ppr_result[target] += source_ppr_vector[ppr_source][bridge] * bridge_ppr_vector[bridge_node][target]

  return ppr_result

def top_ppr(ppr_result):
  sorted_ppr = sorted(ppr_result.items(), key = lambda x:x[1])
  for key, val in sorted_ppr:
    print("Node " + str(key) + " Val " + str(val))
     

def ppr_calc(source, sub_graph, all_graph):
  a = 0.15
  ppr_vector = {}
  ppr_value = 0
  path_value = 1
  path_len = 0
  for target in list(sub_graph.nodes):
    print(target)
    paths = list(nx.all_simple_paths(sub_graph, source=source, target=target))
    for path in paths:
      path_len = len(path) - 1
      if path_len == 0:
        ppr_vector[target] = a * (1-a)
      else:
        for w in path:
          path_value *= 1/all_graph.degree(w)
        path_value *= a * pow(1-a, path_len - 1)
        ppr_value += path_value
        path_value = 1
    ppr_vector[target] = ppr_value
    ppr_value = 0

  return ppr_vector
G = nx.karate_club_graph()
PG = []
objval, fpgaparts = nxmetis.partition(G, int(sys.argv[1]))
# partition into subgraph
for part in fpgaparts:
  PG.append(nx.subgraph(G, part))

# calculate ppr in subgraph
for sub_graph in PG:
  ppr_source = random.choice(list(sub_graph.nodes))
  #print("source " + str(ppr_source))
  #print(nx.pagerank(sub_graph, personalization={ppr_source:1}))
  #nx.draw_networkx(sub_graph)
  #plt.show()

# find bridge edges
bridge_edges = [e for e in G.edges]
for sub_graph in PG:
  sub_edges = []
  for e in sub_graph.edges:
    if e[0] > e[1]:
      sub_edges.append((e[1], e[0]))
    else:
      sub_edges.append(e)
  bridge_edges = list(set(bridge_edges) - set(sub_edges))

# nodes which consists bridge edges
bridge_nodes = set()
for bridge_edge in bridge_edges:
  bridge_nodes.add(bridge_edge[0])
  bridge_nodes.add(bridge_edge[1])

# choose ppr source randomly
#ppr_source = random.choice(list(G.nodes))
ppr_source = 14
source_bridge_nodes = []
bridge_ppr_vector = []
for sub_graph in PG:
  if sub_graph.has_node(ppr_source):
    #source_GM_ppr = nx.pagerank(sub_graph, personalization={ppr_source:1})
    source_bridge_nodes = set(list(sub_graph.nodes)) & bridge_nodes
    source_ppr_vector = source_GM_ppr(sub_graph, ppr_source , source_bridge_nodes)
    #print(source_ppr_vector)
    #print(list(nx.all_simple_paths(sub_graph, source=ppr_source, target=ppr_source)))
    source_ppr_inverse = ppr_calc(ppr_source, sub_graph, G)
  else:
    bridge_ppr_vector.append(bridge_GM_ppr(sub_graph, set(list(sub_graph.nodes)) & bridge_nodes))

print("source : " + str(ppr_source))
merge_ppr = merge(ppr_source, source_bridge_nodes, bridge_edges, source_ppr_inverse, source_ppr_vector, bridge_ppr_vector, G)

print("All GRAPH")
top_ppr(nx.pagerank(G, personalization={ppr_source:1}))
print()
print("Partition GRAPH")
top_ppr(merge_ppr)

node_colors = ['black'] * nx.number_of_nodes(G)
colors = ['red', 'yellow', 'green', 'blue', 'black']
for i, p in enumerate(fpgaparts):
  for node_id in p:
    node_colors[node_id] = colors[i]

#ppr = nx.pagerank(G, personalization={16:1})
#print(ppr)
fig = plt.figure()
#pos = nx.circular_layout(G)
nx.draw_networkx(G, node_color=node_colors)
plt.axis("off")
fig.savefig("partition.png")
