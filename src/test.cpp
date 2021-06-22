#include <iostream>
#include <vector>
#include <unordered_map>
#include <fstream>
#include "graph.hpp"
#include "utils.hpp"

using namespace std;
int main(int argc, char* argv[]) 
{
  if (argc < 3) {
    cerr << "Usage : ./test [graph_path] [partition_num]" << endl;
    exit(1);
  }

  DiGraph all_graph;
  all_graph = read_graph(all_graph, argv[1]);
  int graph_id = 0;
  pair<int, int> num = vertex_edge_num(argv[1]); // vertex_num : edge_num
  cout << "All Graph edge num : " << all_graph.get_num_edges() << endl;
  int tmp = 0;
  for (auto & grpah : partition(argv[1], all_graph.edge_num, stoi(argv[2]))) {
    tmp += grpah.get_num_edges();
    cout << "Graph" << graph_id++ << " -> edges num : " << grpah.get_num_edges() << endl;
  }
  cout << tmp << endl;
  return 0;
}

