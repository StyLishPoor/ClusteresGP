import matplotlib.pyplot as plt
import graphviz
import networkx as nx
import nxmetis
import sys
import random
import math

def top_ppr(ppr_result, nodes):
  #print(ppr_result)
  sorted_ppr = sorted(ppr_result.items(), key = lambda x:x[1])
  for key, val in sorted_ppr:
    if key in nodes:
      print("Node " + str(key) + " Val " + str(val)[0:8])

def calc_ndcg(perfect, estimate):
  sorted_perfect = sorted(perfect.items(), key = lambda x:x[1])
  perfect_num = 0
  index = 2
  for key, val in sorted_perfect:
    if perfect_num == 0:
      perfect_num += val
    else:
      perfect_num += val / math.log(2, index)
      index += 1
  index = 2
  sorted_estimate = sorted(estimate.items(), key = lambda x:x[1])
  estimate_num = 0
  index = 2
  for key, val in sorted_estimate:
    if estimate_num == 0:
      estimate_num += perfect[key] 
    else:
      estimate_num += perfect[key] / math.log(2, index)
      index += 1
  return estimate_num / perfect_num

  
def get_bridge_nodes(G, PG):
  bridge_nodes = set()
  bridge_edges = [e for e in G.edges]
  for sub_graph in PG:
    sub_edges = []
    for e in sub_graph.edges:
      if e[0] > e[1]:
        sub_edges.append((e[0], e[1]))
      else:
        sub_edges.append(e)
    bridge_edges = list(set(bridge_edges) - set(sub_edges))

    for bridge_edge in bridge_edges:
      bridge_nodes.add(bridge_edge[0])
      bridge_nodes.add(bridge_edge[1])
    print(bridge_nodes)
    return bridge_nodes

def RandomWalk(source, a, N, G, source_G_nodes, souce_bridge_nodes):
  visit_count = {}
  out_graph_count = {}
  #for node in list(G.nodes):
  for node in list(source_G_nodes):
    visit_count[node] = 0
    if node in source_bridge_nodes:
      out_graph_count[node] = {}

  RW_count = 0
  current_node = source
  next_node = source
  while (RW_count < N):
    current_node = next_node
    if (random.random() < a):
      visit_count[current_node] += 1
      next_node= source
      RW_count += 1
    else:
      next_node= random.choice(list(G.neighbors(current_node)))
      if (next_node not in source_G_nodes):
        if (next_node not in out_graph_count[current_node]):
          out_graph_count[current_node][next_node] = 1
        else:
          out_graph_count[current_node][next_node] += 1
        next_node = source
        RW_count += 1
  #print(out_graph_count)
  return visit_count, out_graph_count
      
      
def Bridge_RandomWalk(bridge_nodes, a, N, G, bridge_G_nodes):
  result = {}

  for bridge in bridge_nodes:
    visit_count = {}
    out_graph_counts = {}
    for node in list(G.nodes):
      visit_count[node] = 0
      out_graph_counts[node] = {}
    RW_count = 0
    current_node = bridge
    next_node = bridge
    while (RW_count < N):
      current_node = next_node
      if (random.random() < a):
        visit_count[current_node] += 1
        next_node= bridge
        RW_count += 1
      else:
        next_node= random.choice(list(G.neighbors(current_node)))
        if (next_node not in bridge_G_nodes):
          if (next_node not in out_graph_count[current_node]):
            out_graph_count[current_node][next_node] = 1
          else:
            out_graph_count[current_node][next_node] += 1
          next_node = bridge
          RW_count += 1
    result[bridge] = [visit_count, out_graph_counts]
    #print("-------")
    #print(visit_count)
  return result

def Re_Bridge_RandomWalk(re_rw_num, prev_count, a, G, source_G_nodes):

  for start, remain in re_rw_num.items():
    RW_count = 0
    current_node = start 
    next_node =  start
    while (RW_count < int(remain)):
      current_node = next_node
      if (random.random() < a):
        prev_count[current_node] += 1
        next_node = start
        RW_count += 1
      else:
        next_node= random.choice(list(G.neighbors(current_node)))
        if (next_node not in source_G_nodes):
          next_node = start
          RW_count += 1
  #print(prev_count)

def Back_probability(G, a, N, s_bridges, t_bridges):
  probability = {}

  for bridge in t_bridges:
    probability[bridge] = {}


  for bridge in t_bridges:
    out_graph_counts = {}
    for s_bridge in s_bridges:
      out_graph_counts[s_bridge] = 0
    RW_count = 0
    current_node = bridge
    next_node = bridge
    while (RW_count < N):
      current_node = next_node
      if (random.random() < a):
        next_node= bridge
        RW_count += 1
      else:
        next_node= random.choice(list(G.neighbors(current_node)))
        if (next_node in s_bridges):
          out_graph_counts[next_node] += 1
          next_node = bridge
          RW_count += 1
    for node, count in out_graph_counts.items():
      probability[bridge][node] = count / N

  return probability

def estimate_rw_count(source_graph_random_walk_result, out_graph_count, s_bridges, back_probability):
  # back_probability[M][N] : M にでていった RW が N に戻ってくる確率
  # source_graph_random_walk_result[N] : N で RW が終了する回数
  # out_graph_count[N][M] : N から M に RW が出ていった回数
  # for s_bridge in s_bridges:
    # for t_bridge, count in out_graph_count[s_bridge].items():
      # print(str(s_bridge) + " ~ " + str(t_bridge) + " : " +  str(count) + "prob : " + str(back_probability[t_bridge][s_bridge])) 
      # source_graph_random_walk_result[s_bridge] += 0.15 * count * back_probability[t_bridge][s_bridge]
  re_rw_num = {}
  for node in s_bridges:
    re_rw_num[node] = 0

  for _, counts in out_graph_count.items():
    for t_bridge, count in counts.items():
      for s_bridge, prob in back_probability[t_bridge].items():
        #print(str(t_bridge) + " to " + str(s_bridge) + " " + str(count) + " " + str(prob))
        re_rw_num[s_bridge] += (count * prob)
  #print(re_rw_num)
  return re_rw_num

def Bt_return_prob(Bt, s_to_t_prob, t_to_s_prob, ppr, a):
 local_prob = (ppr - a) / (1 + ppr - a)
 out_prob = 0
 for Bs, prob in t_to_s_prob[Bt].items():
  out_prob += prob * s_to_t_prob[Bs][Bt]
 
 return local_prob + out_prob
    

G = nx.karate_club_graph()
PG = []
objval, fpgaparts = nxmetis.partition(G, int(sys.argv[1]))

ppr_source = 14
source_G_nodes = []
target_G_nodes = []

# partition into subgraph
for part in fpgaparts:
  if (ppr_source in part):
    source_graph = nx.subgraph(G, part)
  else:
    target_graph = nx.subgraph(G, part) 
  PG.append(nx.subgraph(G, part))
  if (ppr_source in part):
    source_G_nodes = part
  else:
    target_G_nodes = part

bridge_nodes = get_bridge_nodes(G, PG)
bridge_result = {}
source_bridge_nodes = []
source_graph_random_walk_result = {}
bridge_graph_random_walk_result = {}
target_bridge_nodes = {}
out_graph_count = {}
back_probability = {}
for part in fpgaparts:
  if ppr_source in part:
    source_bridge_nodes = list(set(bridge_nodes) & set(part))
    #source_bridge_nodes = part
    source_graph_random_walk_result, out_graph_count = RandomWalk(ppr_source, 0.15, int(sys.argv[2]), G, part, source_bridge_nodes)
  else:
    target_bridge_nodes = list(set(bridge_nodes) & set(part))
    back_probability = Back_probability(G, 0.15, int(sys.argv[2]), source_bridge_nodes, part)
    #print(back_probability[2])
    #print(back_probability[9])
    #bridge_result = Bridge_RandomWalk(bridge_nodes & set(part), 0.15, int(sys.argv[2]), G, part)
    #print(bridge_result[0])
    #print(bridge_result[2])

#for bridge in source_bridge_nodes:


#for bridge in source_bridge_nodes:
  #for out, count in out_graph_count[bridge].items():
#back_probability(G, 0.15, int(sys.argv[2]), 
    #print(bridge_result[out][1][bridge])
    #source_graph_random_walk_result[bridge] += 0.15 * count * bridge_result[out][1][bridge] / int(sys.argv[2])
tmp = source_graph_random_walk_result.copy()
for node in tmp.keys():
  tmp[node] /= float(sys.argv[2]) 
top_ppr(tmp, source_G_nodes)
re_rw_num = estimate_rw_count(source_graph_random_walk_result, out_graph_count, source_bridge_nodes, back_probability)
Re_Bridge_RandomWalk(re_rw_num, source_graph_random_walk_result, 0.15, G, source_G_nodes)
for node in source_graph_random_walk_result.keys():
  source_graph_random_walk_result[node] /= float(sys.argv[2]) 

out_graph_probability = {}
for s_bridge, counts in out_graph_count.items():
  probability = {}
  for t_bridge, count in counts.items():
    probability[t_bridge] = count / float(sys.argv[2])
  out_graph_probability[s_bridge] = probability

print(out_graph_probability)
print(back_probability)
#direct_probability = {}
#for t_bridge, prob in back_probability.items():
#  direct_probability[t_bridge] = 1 
#  for s_bridge, prob in prob.items():
#    direct_probability[t_bridge] -= prob
#

target_bridge_pprs = {}
for t_bridge in target_bridge_nodes:
  target_bridge_pprs[t_bridge] = nx.pagerank(target_graph, personalization={t_bridge:1})
target_pprs = {}
print("Debug")
for target in target_G_nodes:
  target_pprs[target] = 0
for t_bridge in target_bridge_nodes:
  for s_bridge in source_bridge_nodes:
    if (G.has_edge(s_bridge, t_bridge)):
      for target in target_G_nodes:
        target_pprs[target] += source_graph_random_walk_result[s_bridge] / 0.15  * 0.85 / len(list(G.neighbors(s_bridge))) * target_bridge_pprs[t_bridge][target] / (1 + Bt_return_prob(target, out_graph_probability, back_probability, target_bridge_pprs[t_bridge][t_bridge], 0.15)) 
        #target_pprs[target] += source_graph_random_walk_result[s_bridge] / 0.15  * 0.85 / len(list(G.neighbors(s_bridge))) * direct_probability[t_bridge] * target_bridge_pprs[t_bridge][target]
        #target_pprs[target] += source_graph_random_walk_result[s_bridge] / 0.15  * 0.85 / len(list(G.neighbors(s_bridge))) * ((1 - target_bridge_pprs[t_bridge][t_bridge]) / (1 + target_bridge_pprs[t_bridge][t_bridge])) * target_bridge_pprs[t_bridge][target]

#print(source_graph_random_walk_result)
#print(target_pprs)
#true_pprs = source_graph_random_walk_result | target_pprs
source_graph_random_walk_result.update(target_pprs)
#true_pprs = dict(source_graph_random_walk_result, **target_pprs)
#print(source_bridge_nodes)
#for node, count in random_walk_result:
  #print(str(node) + " : " + str(count))
#print("#-------#")
#print(source_bridge_nodes)
#print(target_bridge_nodes)
#print(source_G_nodes)
#print("#-------#")

print("All GRAPH")
#top_ppr(nx.pagerank(G, personalization={ppr_source:1}), source_bridge_nodes)
#top_ppr(nx.pagerank(G, personalization={ppr_source:1}), source_G_nodes)
perfect_pprs = nx.pagerank(G, personalization={ppr_source:1})
top_ppr(perfect_pprs, list(G.nodes()))
#print(perfect_pprs)

#top_ppr(source_graph_random_walk_result, source_G_nodes)
#top_ppr(source_graph_random_walk_result, list(G.nodes()))

##print(calc_ndcg(perfect_pprs, source_graph_random_walk_result))

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
