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
    print(bridge, out_graph_count)

  print("PROBBBB")
  print(probability)
  #return visit_counts, probability, out_prob
  #print(sys.getsizeof(visit_counts)+sys.getsizeof(probability))
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
  #print(prev_count)

def Back_probability(G, a, N, s_bridges, t_bridges, graph):
  
  #print("--------")
  #print(s_bridges)
  #print("--------")
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
        re_rw_num[s_bridge] += (count * (prob / (1 - prob)))
        #re_rw_num[s_bridge] += (count * prob)

  return re_rw_num

def Bt_return_prob(Bt, s_to_t_prob, t_to_s_prob, ppr, a):
 #local_prob = (ppr - a) / (1 + ppr - a)
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
 #print(Bt, local_prob + out_prob)
 return local_prob + out_prob
 
def BeyondEstimate(s_to_t_probability, t_to_s_probability, s_to_t_count, t_graph_result, target_nodes, N):
  #print("Beyond5G");
  print(s_to_t_count)
  print("Next")
  #print(t_graph_result)
  #print("plz");
  init_flag = False
  true_rw_visit = {}
  t_bridge_start = {}
  t_to_t = {}
  next_t_to_t = {}
  sum_count = {}
  add_t = {}
  back_sum_probability = {}
  s_back_sum_probability = {}
  return_sum_probability = {} 
  
  for v in t_to_s_probability.keys():
    sum_count[v] = 0
    back_sum_probability[v] = 0
    return_sum_probability[v] = 0
    if(init_flag == False):
      for node in t_graph_result[v].keys():
        true_rw_visit[node] = 0
      init_flag = True
    t_bridge_start[v] = 0
    add_t[v] = 0
    next_t_to_t[v] = 0
    t_to_t[v] = 0

  for v in s_to_t_probability.keys():
    s_back_sum_probability[v] = 0

  for t, probs in t_to_s_probability.items():
    for prob in probs.values():
      back_sum_probability[t] += prob

  for s, probs in s_to_t_probability.items():
    for prob in probs.values():
      s_back_sum_probability[s] += prob

  for t_bridge in t_to_s_probability.keys():
    for s_bridge in s_to_t_probability.keys():
      return_sum_probability[t_bridge] += t_to_s_probability[t_bridge][s_bridge] * s_to_t_probability[s_bridge][t_bridge]


  for counts in s_to_t_count.values():
    for t, count in counts.items():
      #t_bridge_start[t] = (1 - back_sum_probability[t]) * count
      #t_to_t[t] = (1 - back_sum_probability[t]) * count
      t_to_t[t] += count 
      #t_bridge_start[t] += (1 - back_sum_probability[t]) * count
  print("t_to_t")
  print(t_to_t)

  t_bridge_start_initial = 0
  for t, num in t_to_t.items():
     t_bridge_start_initial += num
  print("******")
  print(t_bridge_start_initial)
            

  for s, counts in s_to_t_count.items():
    for t, count in counts.items():
      t_bridge_start[t] += (1 - back_sum_probability[t]) * count / (1 - return_sum_probability[t])
      #t_bridge_start[t] += (1 - back_sum_probability[t]) * count / (1 - return_sum_probability[t])
      #t_bridge_start[t] += (1 - back_sum_probability[t]) * count
  print("FUGA")
  print(t_bridge_start)

  t_bridge_start_initial = 0
  for t, num in t_bridge_start.items():
     t_bridge_start_initial += num
  print("******")
  print(t_bridge_start_initial)

  for t_org, remain in t_to_t.items():
    for s in s_to_t_probability.keys():
      for t_target in t_to_s_probability.keys():
        if t_org != t_target:
          # 帰ってきたRWerのうち(1-back_sum_probability[t_target])がプラスされる
          t_bridge_start[t_target] += (1 - back_sum_probability[t_target]) * remain * t_to_s_probability[t_org][s] * s_to_t_probability[s][t_target]
          #t_bridge_start[t_target] += (1 - back_sum_probability[t_target]) * remain * t_to_s_probability[t_org][s] * s_to_t_probability[s][t_target] * 1.5 
          #t_bridge_start[t_target] += remain * t_to_s_probability[t_org][s] * s_to_t_probability[s][t_target]
        #add_t[t_target] += (1 - back_sum_probability[t_target]) * remain * t_to_s_probability[t_org][s] * s_to_t_probability[s][t_target]
        #next_t_to_t[t_target] += back_sum_probability[t_target] * remain * t_to_s_probability[t_org][s] * s_to_t_probability[s][t_target]
          #t_bridge_start[t_target] += remain * t_to_s_probability[t_org][s] * s_to_t_probability[s][t_target] / (1 - return_sum_probability[t_target])
  
  for t_org, remain in t_to_t.items():
    for s in s_to_t_probability.keys():
      for t_target in t_to_s_probability.keys():
        print(t_org, t_target, s)
        next_t_to_t[t_target] += remain * t_to_s_probability[t_org][s] * s_to_t_probability[s][t_target]

  print("Last t_bridge_start")
  print(t_bridge_start)
  t_bridge_start_total = 0
  for t, num in t_bridge_start.items():
    t_bridge_start_total += num
  print(t_bridge_start_total)
  
  print("t_to_t")
  print(t_to_t)
  print("Next")
  print(next_t_to_t)
  print(rw_remain(next_t_to_t))
  print("back sum")
  print(back_sum_probability)
  print(s_back_sum_probability)

  #for s, counts in s_to_t_count.items():
    #for t, count in counts.items():
      #t_bridge_start[t] += counts[t] * ((1 - back_sum_probability[t] + back_sum_probability[t] * return_sum_probability[t]) / (back_sum_probability[t] - back_sum_probability[t] * return_sum_probability[t]))
      #t_bridge_start[t] += counts[t] * (1 - 0.75*back_sum_probability[t])
      #t_bridge_start[t] += s_to_t_count[s][t] * (1 - probs[s])
    #for t, count in counts.items():
      # やや低く見積もられた値が計算されてしまう．RW １万回のとき，約1.3倍で良さげな値になる
      #t_bridge_start[t] += count / (1 - s_to_t_probability[s][t]*t_to_s_probability[t][s]) 
      #t_bridge_start[t] += (1 - back_sum_probability[t]) * count / (1 - s_to_t_probability[s][t]*t_to_s_probability[t][s]) 
      #t_bridge_start[t] += (1 - back_sum_probability[t]) * count / (1 - return_sum_probability[t])
      #t_bridge_start[t] += (1 - back_sum_probability[t]) * count 
      #t_bridge_start[t] += (1 - t_to_s_probability[t][s]) * count / (1 - s_to_t_probability[s][t]*t_to_s_probability[t][s]) 
      #t_bridge_start[t] += (1 - return_sum_probability[t]) * count / (1 - s_to_t_probability[s][t]*t_to_s_probability[t][s]) 
      #t_bridge_start[t] += count / (1 - 3*s_to_t_probability[s][t]*3*t_to_s_probability[t][s]) 
      #t_bridge_start[t] += count
      #sum_count[t] += count
      #print(str(s) + " <-> " + str(t) + " : " + str(s_to_t_probability[s][t]) + " " + str(t_to_s_probability[t][s]))
      #t_bridge_start[t] += count / (1 - s_to_t_probability[s][t]*t_to_s_probability[t][s]) 
      #t_bridge_start[t] += count
  #for node in sum_count.keys():
  #  print(t_bridge_start[node] / sum_count[node])

  # 初期化
  for t in t_to_t.keys():
    t_bridge_start[t] = 0
    next_t_to_t[t] = 0

  print("NoraMethod!!!!")
  
  #for s, counts in s_to_t_count.items():
    #for t, count in counts.items():
      #t_bridge_start[t] += (1 - back_sum_probability[t]) * count / (1 - return_sum_probability[t])
  #for t_org, remain in t_to_t.items():
    #for s in s_to_t_probability.keys():
      #for t_target in t_to_s_probability.keys():
        #if t_org != t_target:
          ## 帰ってきたRWerのうち(1-back_sum_probability[t_target])がプラスされる
          #t_bridge_start[t_target] += (1 - back_sum_probability[t_target]) * remain * t_to_s_probability[t_org][s] * s_to_t_probability[s][t_target]
  #t_bridge_start_total = 0

  # t_to_tとnext_t_to_tを使って残りが閾値を下回るまで回し続けるパワープレー
  while(rw_remain(t_to_t) > 1):
    for t_org, remain in t_to_t.items():
      # まずは現在の残りから(1-back_sum_probability)だけ確定させる
      t_bridge_start[t_org] += (1 - back_sum_probability[t_org]) * remain
      print("Hoge")
      print(t_bridge_start)
      for s in s_to_t_probability.keys():
        for t_target in t_to_s_probability.keys():
          next_t_to_t[t_target] += remain * t_to_s_probability[t_org][s] * s_to_t_probability[s][t_target]
    # next_t_to_tをt_to_tに代入し，next_t_to_tを初期化
    print("CHECK")
    print(next_t_to_t)
    print(rw_remain(next_t_to_t))
    for t in t_to_t.keys():
      t_to_t[t] = next_t_to_t[t]
      next_t_to_t[t] = 0
  print("NoraMethod")
  t_bridge_start_total = 0
  for t, num in t_bridge_start.items():
    t_bridge_start_total += num
  print(t_bridge_start_total)

  # 各tの変化率だけから求めようと試みた
  # 変化率は他tで残っているRWerの回数に依存するためうまくいかない
  #for t_org, remain in t_to_t.items():
    #for s in s_to_t_probability.keys():
      #for t_target in t_to_s_probability.keys():
        #next_t_to_t[t_target] += remain * t_to_s_probability[t_org][s] * s_to_t_probability[s][t_target]

  for t, t_results in t_graph_result.items():
    for target, count in t_results.items():
      true_rw_visit[target] += count * (t_bridge_start[t]/N) / N 
      #true_rw_visit[target] += count * (t_bridge_start[t]/N) / N 
  return true_rw_visit 

def rw_remain(t_to_t):
  remain_total = 0
  for num in t_to_t.values():
    remain_total += num
  return remain_total

G = nx.karate_club_graph()
print(sys.getsizeof(G))
print(len(list(G.edges)))
#G = nx.read_edgelist("./dataset/web-Stanford.txt", comments='#')
PG = []
objval, fpgaparts = nxmetis.partition(G, int(sys.argv[1]))
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
    print("SOURCE")
    print(source_graph_random_walk_result)
    #beyond_probability = Back_probability(G, 0.15, int(sys.argv[2]), target_bridge_nodes, source_bridge_nodes, part)
    #beyond_probability = Back_probability(G, 0.15, int(sys.argv[2]), target_bridge_nodes, source_bridge_nodes)
    s_bridge_results, beyond_probability = Bridge_RandomWalk(G, part, 0.15, int(sys.argv[2]), target_bridge_nodes, source_bridge_nodes)
    print(beyond_probability)

  else:
    t_bridge_results, back_probability = Bridge_RandomWalk(G, part, 0.15, int(sys.argv[2]), source_bridge_nodes, target_bridge_nodes)
    print(back_probability)

t_graph_pprs = BeyondEstimate(beyond_probability, back_probability, out_graph_count, t_bridge_results, target_G_nodes, int(sys.argv[2]))

tmp = source_graph_random_walk_result.copy()
for node in tmp.keys():
  tmp[node] /= float(sys.argv[2]) 
#top_ppr(tmp, source_G_nodes)
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

#direct_probability = {}
#for t_bridge, prob in back_probability.items():
#  direct_probability[t_bridge] = 1 
#  for s_bridge, prob in prob.items():
#    direct_probability[t_bridge] -= prob
#


#target_bridge_pprs = {}
#for t_bridge in target_bridge_nodes:
#  target_bridge_pprs[t_bridge] = nx.pagerank(target_graph, personalization={t_bridge:1})
#target_pprs = {}
#print("Debug")
#for target in target_G_nodes:
#  target_pprs[target] = 0
#for t_bridge in target_bridge_nodes:
#  for s_bridge in source_bridge_nodes:
#    if (G.has_edge(s_bridge, t_bridge)):
#      for target in target_G_nodes:
        #target_pprs[target] += source_graph_random_walk_result[s_bridge] / 0.15  * 0.85 / len(list(G.neighbors(s_bridge))) * target_bridge_pprs[t_bridge][target] / ((target_bridge_pprs[t_bridge][t_bridge] / 0.15) + (1 / 1 -  Bt_return_prob(t_bridge, out_graph_probability, back_probability, target_bridge_pprs[t_bridge][t_bridge], 0.15)))
        #target_pprs[target] += source_graph_random_walk_result[s_bridge] / 0.15  * 0.85 / len(list(G.neighbors(s_bridge))) * target_bridge_pprs[t_bridge][target] /((target_bridge_pprs[t_bridge][t_bridge] / 0.15));
        #target_pprs[target] += source_graph_random_walk_result[s_bridge] / 0.15  * 0.85 / len(list(G.neighbors(s_bridge))) * target_bridge_pprs[t_bridge][target] / ((1/1-(Bt_return_prob(t_bridge, out_graph_probability, back_probability, target_bridge_pprs[t_bridge][t_bridge], 0.15))) + (target_bridge_pprs[t_bridge][t_bridge] / 0.15));
        #target_pprs[target] += source_graph_random_walk_result[s_bridge] / 0.15  * 0.85 / len(list(G.neighbors(s_bridge))) * target_bridge_pprs[t_bridge][target] / ((1 + target_graph.degree(target)/G.degree(target)) * target_bridge_pprs[t_bridge][t_bridge] / 0.15);
        #target_pprs[target] += source_graph_random_walk_result[s_bridge] / 0.15  * 0.85 / len(list(G.neighbors(s_bridge))) * (target_graph.degree(t_bridge) / len(list(G.neighbors(t_bridge)))) * target_bridge_pprs[t_bridge][target]
        #target_pprs[target] += source_graph_random_walk_result[s_bridge] / 0.15  * 0.85 / len(list(G.neighbors(s_bridge))) * target_bridge_pprs[t_bridge][target] * (1 - 5 * Bt_return_prob(t_bridge, out_graph_probability, back_probability, target_bridge_pprs[t_bridge][t_bridge], 0.15)) 
        #target_pprs[target] += source_graph_random_walk_result[s_bridge] / 0.15  * 0.85 / len(list(G.neighbors(s_bridge))) * direct_probability[t_bridge] * target_bridge_pprs[t_bridge][target]
        #target_pprs[target] += source_graph_random_walk_result[s_bridge] / 0.15  * 0.85 / len(list(G.neighbors(s_bridge))) * ((1 - target_bridge_pprs[t_bridge][t_bridge]) / (1 + target_bridge_pprs[t_bridge][t_bridge])) * target_bridge_pprs[t_bridge][target]

#print(source_graph_random_walk_result)
#print(target_pprs)
#source_graph_random_walk_result.update(target_pprs)
source_graph_random_walk_result.update(t_graph_pprs)
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
#top_ppr(nx.pagerank(G, personalization={ppr_source:1}), target_G_nodes)
perfect_pprs = nx.pagerank(G, personalization={ppr_source:1})
total_t_visit = 0
total_s_visit = 0
for node, value in perfect_pprs.items():
  if node in source_G_nodes:
    total_s_visit += value
  if node in target_G_nodes:
    total_t_visit += value
print("Total visit S")
print(total_s_visit * int(sys.argv[2]))
print("Total visit T")
print(total_t_visit * int(sys.argv[2]))

top_ppr(perfect_pprs, list(G.nodes()))
#print(perfect_pprs)

print("nora GRAPH")
#top_ppr(source_graph_random_walk_result, source_G_nodes)
top_ppr(source_graph_random_walk_result, list(G.nodes()))
#top_ppr(source_graph_random_walk_result, list(target_G_nodes))

print(calc_ndcg(perfect_pprs, source_graph_random_walk_result))

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
#fig.savefig("partition.png")
