#ifndef GRAPH
#define GRAPH
#include <iostream>
//#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
//#include <stdlib.h>

using namespace std;
using Vertex = unsigned int;
using Edge = pair<Vertex, Vertex>;

struct DiGraph {
  unordered_set<Vertex> vertices;
  unordered_map<Vertex, vector<Vertex>> adjacency_list;
  unsigned int edge_num;

  DiGraph() {
    edge_num = 0;
  }
  
  void add_edge(Vertex s, Vertex t);
  void add_vertex(Vertex v);
  bool has_vertex(Vertex v);
  vector<Vertex> get_vertices();
  int get_num_vertices();
  int get_num_edges();
  int get_degree(Vertex v);
  vector<Vertex> get_neighbors(Vertex v);
  unsigned int max_id();
};

#endif
