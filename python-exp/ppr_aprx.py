import matplotlib.pyplot as plt
import graphviz
import networkx as nx
#from networkx.algorithms import community
import community as community_louvain
import nxmetis
import sys
import os
import random
import math
import time
import json

def top_ppr(ppr_result, nodes, top_k):
  sorted_ppr = sorted(ppr_result.items(), key = lambda x:x[1], reverse=True)
  count = 0
  for key, val in sorted_ppr:
    count += 1
    if key in nodes:
      if count < top_k: 
        print("Node " + str(key) + " Val " + str(val)[0:8])

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def calc_ndcg(perfect, estimate):
  sorted_perfect = sorted(perfect.items(), key = lambda x:x[1], reverse=True)
  perfect_num = -1
  index = 2
  for key, val in sorted_perfect:
    if perfect_num == -1:
      perfect_num = val
    else:
      perfect_num += val / math.log2(index)
      index += 1
  index = 2
  sorted_estimate = sorted(estimate.items(), key = lambda x:x[1], reverse=True)
  estimate_num = -1 
  index = 2
  for key, val in sorted_estimate:
    if estimate_num == -1:
      estimate_num = perfect[key] 
    else:
      estimate_num += perfect[key] / math.log2(index)
      index += 1

  return estimate_num / perfect_num

def calc_data_size(receive_bridge, send_bridge, t_results, t_back):

  send_results = {}
  send_back = {}
  for bridge in send_bridge:
    send_results[bridge] = t_results[bridge]
    send_back[bridge] = t_back[bridge]

  file1 = open('json/result.json', 'w')
  file2 = open('json/prob.json', 'w')
  json.dump(send_results, file1)
  json.dump(send_back, file2)
  
  return os.path.getsize('json/result.json'), os.path.getsize('json/prob.json')

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

  file1 = open('json/source-rw.json', 'w')
  file2 = open('json/out-rw.json', 'w')
  json.dump(visit_count, file1)
  json.dump(out_graph_count, file2)
  return visit_count, out_graph_count
      
      
def Bridge_RandomWalk(G, Sub_G, a, N, out_bridges, in_bridges):
  visit_counts = {} # visit_counts[N][M] : Bridge N ??? M ?????????
  probability = {} # probability[N][M] : N ??? M ?????????????????????
  out_prob = {} # N ??? ????????????????????????

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
        #if (next_node not in out_bridges):
        if (next_node not in Sub_G):
          if next_node not in out_graph_count:
            out_graph_count[next_node] = 0
          out_graph_count[next_node] += 1
          out_count += 1
          next_node = bridge
          RW_count += 1



    tmp_count = {}
    for target, count in visit_count.items():
      tmp_count[target] = int(count * N / (N - out_count))

    visit_counts[bridge] = tmp_count

    for node, count in out_graph_count.items():
      probability[bridge][node] = count / N
    
  file1 = open('json/source-count.json', 'w')
  file2 = open('json/source-probability.json', 'w')
  json.dump(visit_counts, file1)
  json.dump(probability, file2)

  return visit_counts, probability

def Bridge_RandomWalk2(G, Sub_G, a, N, out_bridges, in_bridges):
  json_count = {} 
  visit_counts = {} # visit_counts[N][M] : Bridge N ??? M ?????????
  probability = {} # probability[N][M] : N ??? M ?????????????????????
  out_prob = {} # N ??? ????????????????????????

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
        #if (next_node not in out_bridges):
        if (next_node not in Sub_G):
          if next_node not in out_graph_count:
            out_graph_count[next_node] = 0
          out_graph_count[next_node] += 1
          out_count += 1
          next_node = bridge
          RW_count += 1



    tmp_count = {}
    tmp2_count = {}
    for target, count in visit_count.items():
      tmp2_count[target] = int(count * N / (N - out_count))
      if count >= 20:
        tmp_count[target] = int(count * N / (N - out_count))

    json_count[bridge] = tmp2_count
    visit_counts[bridge] = tmp_count

    for node, count in out_graph_count.items():
      probability[bridge][node] = count / N
    

  file1 = open('json/target-count.json', 'w')
  file2 = open('json/target-probability.json', 'w')
  json.dump(json_count, file1)
  json.dump(probability, file2)

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
  # back_probability[M][N] : M ?????????????????? RW ??? N ????????????????????????
  # source_graph_random_walk_result[N] : N ??? RW ?????????????????????
  # out_graph_count[N][M] : N ?????? M ??? RW ????????????????????????

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
 
def decide_skip(bridge_nodes, G, p):
  degree = {}
  for bridge in bridge_nodes:
    degree[bridge] = G.degree(bridge)
  sort_degree = sorted(degree.items(), key = lambda x:x[1])
  thr = int((len(bridge_nodes) * p))
  skip = []
  for bridge in sort_degree:
    if len(skip) < thr:
      skip.append(bridge[0])

  return skip

def read_json(count_path, probability_path, source=True):
  count_file = open(count_path, 'r')
  probability_file = open(probability_path, 'r')

  if source:
    source_count = json.load(count_file)
    source_probability = json.load(probability_file)
    return source_count, source_probability
  else:
    target_count = json.load(count_file)
    target_probability = json.load(probability_file)
    return target_count, target_probability

   
def BeyondEstimate(s_to_t_probability, t_to_s_probability, s_to_t_count, t_graph_result, target_nodes, s_skip, t_skip, N):
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
      if s not in s_skip:
        s_back_sum_probability[s] += prob

  for counts in s_to_t_count.values():
    for t, count in counts.items():
      if t in t_to_s_probability.keys() and t not in t_skip:
        t_to_t[t] += count 

  # ????????????????????????????????????????????????
  while(rw_remain(t_to_t) > 1):
    for t_org, remain in t_to_t.items():
      if t_org not in t_skip:
        # ??????????????????????????????(1-back_sum_probability)?????????????????????
        t_bridge_start[t_org] += (1 - back_sum_probability[t_org]) * remain
        for s in s_to_t_probability.keys():
          if s not in s_skip and s in t_to_s_probability[t_org].keys():
            s_bridge_re_start[s] += (1 - s_back_sum_probability[s]) * remain * t_to_s_probability[t_org][s] 
            for t_target in t_to_s_probability.keys():
              if t_target not in t_skip and s in t_to_s_probability[t_org].keys() and t_target in s_to_t_probability[s].keys():
                next_t_to_t[t_target] += remain * t_to_s_probability[t_org][s] * s_to_t_probability[s][t_target]
    # next_t_to_t???t_to_t???????????????next_t_to_t????????????
    for t in t_to_t.keys():
      if t not in t_skip:
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
  s_result = source_graph_random_walk_result

  for t in target_nodes:
    t_result[t] = 0

  #for s, visit in source_graph_random_walk_result.items():
    #s_result[s] = visit / N

  for t, t_results in t_bridge_result.items():
    for target, count in t_results.items():
      t_result[target] += count * (t_bridge_start[t]/N) / N 
   
  for s, s_results in s_bridge_result.items():
    for target, count in s_results.items():
      s_result[target] += count * (s_bridge_re_start[s]/N) / N 
  
  s_result.update(t_result)
  return s_result

#G = nx.karate_club_graph()
#G = nx.fast_gnp_random_graph(1000, 0.1)
#G = nx.read_edgelist("./dataset/web-Stanford.txt", comments='#')
G = nx.read_edgelist("./dataset/email-Eu-core.txt", nodetype=int, comments='#')
size_all = os.path.getsize("./dataset/email-Eu-core.txt")
print(size_all)
#communities_generator = community.girvan_newman(G)
#next_level_communities = next(communities_generator)
#G = nx.read_edgelist("./dataset/p2p-Gnutella08.txt", comments='#')

partition = community_louvain.best_partition(G)
partitioned_nodes = {}
for node, community in partition.items():
  if not  community in partitioned_nodes:
    partitioned_nodes[community] = []
  partitioned_nodes[community].append(int(node))


PG = []
#objval, fpgaparts = nxmetis.partition(G, int(sys.argv[1]))
objval, partitioned_nodes = nxmetis.partition(G, int(sys.argv[1]))

ppr_source = 14
source_G_nodes = []
target_G_nodes = []

# partition into subgraph
#for part in fpgaparts:
for part in partitioned_nodes:
#for part in partitioned_nodes.values():
  PG.append(nx.subgraph(G, part))
  if (ppr_source in part):
    source_G_nodes = part
  else:
    target_G_nodes.append(part)

bridge_nodes = get_bridge_nodes(G, PG)
bridge_result = {}
result = {}
source_bridge_nodes = []
target_bridge_nodes = []
source_graph_random_walk_result = {}
bridge_graph_random_walk_result = {}
out_graph_count = {}
back_probability = {}
beyond_probability = {}
#for part in fpgaparts:
source_bridge_nodes = list(set(bridge_nodes) & set(source_G_nodes))
all_target_bridge_nodes = []
target_bridge_nodes = []
for nodes in target_G_nodes:
  all_target_bridge_nodes.extend(list(set(bridge_nodes) & set(nodes)))

#for part in partitioned_nodes.values():
  #if ppr_source in part:
    #source_bridge_nodes = list(set(bridge_nodes) & set(part))
  #else:
    #target_bridge_nodes = list(set(bridge_nodes) & set(part))

# source?????????RW
#source_graph_random_walk_result, out_graph_count = RandomWalk(ppr_source, 0.15, int(sys.argv[2]), G, source_G_nodes, source_bridge_nodes)
source_graph_random_walk_result_json, out_graph_count_json = read_json("json/source-rw.json", "json/out-rw.json")
for s, count in source_graph_random_walk_result_json.items():
  source_graph_random_walk_result[int(s)] = int(count)
for s, counts in out_graph_count_json.items():
  out_graph_count[int(s)] = {}
  for target, count in counts.items():
    out_graph_count[int(s)][int(target)] = int(count)
#s_bridge_results, beyond_probability = Bridge_RandomWalk(G, source_G_nodes, 0.15, int(sys.argv[2]), all_target_bridge_nodes, source_bridge_nodes)
s_bridge_results_json, beyond_probability_json = read_json("json/source-count.json", "json/source-probability.json")
#s_bridge_results = {int(k): int(v) for k, v in s_bridge_results_json.items()}

s_bridge_results = {}
beyond_probability = {}
for s, results in s_bridge_results_json.items():
  s_bridge_results[int(s)] = {}
  for target, count in results.items():
    s_bridge_results[int(s)][int(target)] = int(count)

for s, probs in beyond_probability_json.items():
  beyond_probability[int(s)] = {}
  for target, prob in probs.items():
    beyond_probability[int(s)][int(target)] = float(prob)

for s, visit in source_graph_random_walk_result.items():
  result[s] = visit / int(sys.argv[2])

# target?????????RW
for part in target_G_nodes:
  target_bridge_nodes = list(set(bridge_nodes) & set(part))
  #t_bridge_results, back_probability = Bridge_RandomWalk2(G, part, 0.15, int(sys.argv[2]), source_bridge_nodes, target_bridge_nodes)
  t_bridge_results_json, back_probability_json = read_json("json/target-count.json", "json/target-probability.json", False)
  t_bridge_results = {}
  back_probability = {}
  for t, results in t_bridge_results_json.items():
    t_bridge_results[int(t)] = {}
    for target, count in results.items():
      if int(count) > int(sys.argv[5]):
        t_bridge_results[int(t)][int(target)] = int(count)

  for t, probs in back_probability_json.items():
    back_probability[int(t)] = {}
    for target, prob in probs.items():
      if float(prob) > float(sys.argv[6]):
        back_probability[int(t)][int(target)] = float(prob)

  s_skip = decide_skip(source_bridge_nodes, G, float(sys.argv[3]))
  t_skip = decide_skip(target_bridge_nodes, G, float(sys.argv[4]))
  s_receive_bridge = list(set(source_bridge_nodes) - set(s_skip))
  t_send_bridge = list(set(target_bridge_nodes) - set(t_skip))
  t_bridge_start, s_bridge_re_start = BeyondEstimate(beyond_probability, back_probability, out_graph_count, t_bridge_results, target_G_nodes, s_skip, t_skip, int(sys.argv[2]))

  result = merge_rw(result, t_bridge_results, s_bridge_results, t_bridge_start, s_bridge_re_start, part, int(sys.argv[2]))

#merged_ppr = merge_rw(source_graph_random_walk_result, t_bridge_results, s_bridge_results, t_bridge_start, s_bridge_re_start, target_G_nodes, int(sys.argv[2]))
size1, size2 = calc_data_size(s_receive_bridge, t_send_bridge, t_bridge_results, back_probability)
print("start")
top_ppr(result, list(G.nodes), 50)
check = 0
for val in result.values():
  check += val
print(check)

print("Compare")
perfect_pprs = nx.pagerank(G, personalization={ppr_source:1})
top_ppr(perfect_pprs, list(G.nodes()), 50)
ndcg = calc_ndcg(perfect_pprs, result)
size1 = os.path.getsize('json/result.json')
size2 = os.path.getsize('json/prob.json')
output_file = open("nora.txt", 'a')
output_file.write(str(size1) + " "  + str(size2) + " " +  str(100*(size1+size2)/size_all)+ " " +  str(check) + " " +  str(ndcg))
output_file.close()

#node_colors = ['black'] * nx.number_of_nodes(G)
#colors = ['red', 'yellow', 'green', 'blue', 'black']
#for i, p in enumerate(partitioned_nodes.values()):
  #for node_id in p:
    #node_colors[node_id] = colors[i]

#ppr = nx.pagerank(G, personalization={16:1})
#print(ppr)
#fig = plt.figure()
#pos = nx.circular_layout(G)
#nx.draw_networkx(G, node_color=node_colors)
#plt.axis("off")
#fig.savefig("partition.png")
