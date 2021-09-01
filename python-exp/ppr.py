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
  return bridge_nodes

 
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

def main():
  G = nx.karate_club_graph()
#G = nx.read_edgelist("./dataset/web-Stanford.txt", comments='#')
  print("READ")
  PG = []
  objval, fpgaparts = nxmetis.partition(G, int(sys.argv[1]))
  print("PARTITION")
#print(fpgaparts)

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
      #beyond_probability = Back_probability(G, 0.15, int(sys.argv[2]), target_bridge_nodes, source_bridge_nodes, part)
      #beyond_probability = Back_probability(G, 0.15, int(sys.argv[2]), target_bridge_nodes, source_bridge_nodes)
      s_bridge_results, beyond_probability = Bridge_RandomWalk(G, part, 0.15, int(sys.argv[2]), target_bridge_nodes, source_bridge_nodes)

    else:
      t_bridge_results, back_probability = Bridge_RandomWalk(G, part, 0.15, int(sys.argv[2]), source_bridge_nodes, target_bridge_nodes)

# t_graph_pprs = BeyondEstimate(beyond_probability, back_probability, out_graph_count, t_bridge_results, target_G_nodes, int(sys.argv[2]))

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

  '
  node_colors = ['black'] * nx.number_of_nodes(G)
  colors = ['red', 'yellow', 'green', 'blue', 'black']
  for i, p in enumerate(fpgaparts):
    for node_id in p:
      node_colors[node_id] = colors[i]

  fig = plt.figure()
  nx.draw_networkx(G, node_color=node_colors)
  plt.axis("off")
  ' 

if __name__ == '__main__':
  main()
