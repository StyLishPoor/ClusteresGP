#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "graph.hpp"

void DiGraph::add_edge(Vertex s, Vertex t) {
  adjacency_list[s].push_back(t);
}

void DiGraph::add_vertex(Vertex v) {
  vertices.insert(v);
}

bool DiGraph::has_vertex(Vertex v) {
  if (vertices.find(v) == vertices.end()) {
    return false;
  }
  return true;
}

vector<Vertex> DiGraph::get_vertices() {
  vector<Vertex> all_vertices(vertices.begin(), vertices.end());
  return all_vertices;
}

int DiGraph::get_num_vertices() {
  return vertices.size();
}

int DiGraph::get_num_edges() {
  int edge_num = 0;
  for (auto & adj : adjacency_list) {
    edge_num += adj.second.size();
  }
  return edge_num;
}

int DiGraph::get_degree(Vertex v) {
  return adjacency_list[v].size();
}

vector<Vertex> DiGraph::get_neighbors(Vertex v) {
  return adjacency_list[v];
}

unsigned int DiGraph::max_id() {
  vector<Vertex> node_list = get_vertices();
  sort(node_list.begin(), node_list.end());
  return node_list[node_list.size() - 1];
}
