import matplotlib.pyplot as plt
import graphviz
import networkx as nx
import community as community_louvain
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
  return visit_count, out_graph_count
      
      
def Bridge_RandomWalk(G, Sub_G, a, N, out_bridges, in_bridges):
  visit_counts = {} # visit_counts[N][M] : Bridge N → M の回数
  probability = {} # probability[N][M] : N → M にでていく確率
  out_prob = {} # N → 外に出ていく確率


  for bridge in in_bridges:
    probability[bridge] = {}
    visit_count = {}
    out_graph_count = {}
    
    for node in Sub_G:
      visit_count[node] = 0
    for node in out_bridges:
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
        next_node= random.choice(list(G.neighbors(current_node)))
        if (next_node in out_bridges):
          out_graph_count[next_node] += 1
          out_count += 1
          next_node = bridge
          RW_count += 1



    for target, count in visit_count.items():
      visit_count[target] = count * N / (N - out_count)
    
    for node, count in out_graph_count.items():
      probability[bridge][node] = count / N
    
    visit_counts[bridge] = visit_count

  return visit_counts, probability

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

def Back_probability(G, a, N, s_bridges, t_bridges, graph):
  
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

  re_rw_num = {}
  for node in s_bridges:
    re_rw_num[node] = 0

  for _, counts in out_graph_count.items():
    for t_bridge, count in counts.items():
      for s_bridge, prob in back_probability[t_bridge].items():
        re_rw_num[s_bridge] += (count * (prob / (1 - prob)))

  return re_rw_num

def Bt_return_prob(Bt, s_to_t_prob, t_to_s_prob, ppr, a):
 out_prob = 0
 for Bs, prob in t_to_s_prob[Bt].items():
  if Bt in s_to_t_prob[Bs]:
    out_prob += prob * s_to_t_prob[Bs][Bt]
 return out_prob
    

def Bt_return_num(ppr, a):
 local_prob = (ppr - a) / (1 + ppr - a)
 out_prob = 0
 for Bs, prob in t_to_s_prob[Bt].items():
  if Bt in s_to_t_prob[Bs]:
    out_prob += prob * s_to_t_prob[Bs][Bt]
 return local_prob + out_prob
 
def BeyondEstimate(s_to_t_probability, t_to_s_probability, s_to_t_count, t_graph_result, target_nodes, N):
  t_bridge_start = {}
  s_bridge_re_start = {}

  t_to_t = {}
  s_to_s = {}
  next_t_to_t = {}
  next_s_to_s = {}

  back_sum_probability = {}
  s_back_sum_probability = {}

  return_sum_probability = {} 
  
  for v in t_to_s_probability.keys():
    back_sum_probability[v] = 0
    t_bridge_start[v] = 0
    next_t_to_t[v] = 0
    t_to_t[v] = 0

  for v in s_to_t_probability.keys():
    s_back_sum_probability[v] = 0
    s_bridge_re_start[v] = 0
    next_s_to_s[v] = 0
    s_to_s[v] = 0

  for t, probs in t_to_s_probability.items():
    for prob in probs.values():
      back_sum_probability[t] += prob

  for s, probs in s_to_t_probability.items():
    for prob in probs.values():
      s_back_sum_probability[s] += prob

  for counts in s_to_t_count.values():
    for t, count in counts.items():
      t_to_t[t] += count 

  # 割り振りが終了するまで回し続ける
  while(rw_remain(t_to_t) > 1):
    for t_org, remain in t_to_t.items():
      # まずは現在の残りから(1-back_sum_probability)だけ確定させる
      t_bridge_start[t_org] += (1 - back_sum_probability[t_org]) * remain
      for s in s_to_t_probability.keys():
        s_bridge_re_start[s] += (1 - s_back_sum_probability[s]) * remain * t_to_s_probability[t_org][s] 
        for t_target in t_to_s_probability.keys():
          next_t_to_t[t_target] += remain * t_to_s_probability[t_org][s] * s_to_t_probability[s][t_target]
    # next_t_to_tをt_to_tに代入し，next_t_to_tを初期化
    for t in t_to_t.keys():
      t_to_t[t] = next_t_to_t[t]
      next_t_to_t[t] = 0

  return t_bridge_start, s_bridge_re_start

def rw_remain(t_to_t):
  remain_total = 0
  for num in t_to_t.values():
    remain_total += num
  return remain_total

def merge_rw(source_graph_random_walk_result, t_bridge_result, s_bridge_result, t_bridge_start, s_bridge_re_start, target_nodes, N):
  t_result = {}
  s_result = {}

  for t in target_nodes:
    t_result[t] = 0

  for s, visit in source_graph_random_walk_result.items():
    s_result[s] = visit / N

  for t, t_results in t_bridge_result.items():
    for target, count in t_results.items():
      t_result[target] += count * (t_bridge_start[t]/N) / N 
   
  for s, s_results in s_bridge_result.items():
    for target, count in s_results.items():
      s_result[target] += count * (s_bridge_re_start[s]/N) / N 
  
  s_result.update(t_result)
  return s_result

G = nx.karate_club_graph()
#G = nx.read_edgelist("./dataset/web-Stanford.txt", comments='#')
#G = nx.read_edgelist("./dataset/email-Eu-core.txt", comments='#')
#communities_generator = community.girvan_newman(G)
#next_level_communities = next(communities_generator)
#G = nx.read_edgelist("./dataset/p2p-Gnutella08.txt", comments='#')
PG = []
objval, fpgaparts = nxmetis.partition(G, int(sys.argv[1]))
print("PARTITION")
#print(fpgaparts)

ppr_source = 14
source_G_nodes = []
target_G_nodes = []

# partition into subgraph
for part in fpgaparts:
# for part in partitioned_nodes.values():
  print(part)
  PG.append(nx.subgraph(G, part))
  if (ppr_source in part):
    source_G_nodes = part
  else:
    target_G_nodes = part

bridge_nodes = get_bridge_nodes(G, PG)
print(len(bridge_nodes))
bridge_result = {}
source_bridge_nodes = []
target_bridge_nodes = []
source_graph_random_walk_result = {}
bridge_graph_random_walk_result = {}
out_graph_count = {}
back_probability = {}
beyond_probability = {}
for part in fpgaparts:
  if ppr_source in part:
    source_bridge_nodes = list(set(bridge_nodes) & set(part))
  else:
    target_bridge_nodes = list(set(bridge_nodes) & set(part))

for part in fpgaparts:
  if ppr_source in part:
    #source_bridge_nodes = part
    source_graph_random_walk_result, out_graph_count = RandomWalk(ppr_source, 0.15, int(sys.argv[2]), G, part, source_bridge_nodes)
    s_bridge_results, beyond_probability = Bridge_RandomWalk(G, part, 0.15, int(sys.argv[2]), target_bridge_nodes, source_bridge_nodes)

  else:
    t_bridge_results, back_probability = Bridge_RandomWalk(G, part, 0.15, int(sys.argv[2]), source_bridge_nodes, target_bridge_nodes)

t_bridge_start, s_bridge_re_start = BeyondEstimate(beyond_probability, back_probability, out_graph_count, t_bridge_results, target_G_nodes, int(sys.argv[2]))

merged_ppr = merge_rw(source_graph_random_walk_result, t_bridge_results, s_bridge_results, t_bridge_start, s_bridge_re_start, target_G_nodes, int(sys.argv[2]))
print("start")
top_ppr(merged_ppr, list(G.nodes))
check = 0
for val in merged_ppr.values():
  check += val
print(check)

print("Compare")
perfect_pprs = nx.pagerank(G, personalization={ppr_source:1})
top_ppr(perfect_pprs, list(G.nodes()))
print(calc_ndcg(perfect_pprs, merged_ppr))

#node_colors = ['black'] * nx.number_of_nodes(G)
#colors = ['red', 'yellow', 'green', 'blue', 'black']
#for i, p in enumerate(fpgaparts):
  #for node_id in p:
    #node_colors[node_id] = colors[i]

#ppr = nx.pagerank(G, personalization={16:1})
#print(ppr)
#fig = plt.figure()
#pos = nx.circular_layout(G)
#nx.draw_networkx(G, node_color=node_colors)
#plt.axis("off")
#fig.savefig("partition.png")
