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

  ifstream ifs(argv[1]);
  if (!ifs) {
    cerr << "Cannot open file" << endl;
    exit(1);
  }

  DiGraph all_graph;
  all_graph = read_graph(all_graph, ifs);
  cout << all_graph.edge_num << endl;
  int graph_id = 0;
  pair<int, int> num = vertex_edge_num(ifs); // vertex_num : edge_num
  for (auto & grpah : partition(ifs, all_graph.edge_num, stoi(argv[2]))) {
    cout << "Hello" << endl;
    cout << "Graph" << graph_id++ << " -> max id : " << grpah.get_num_vertices() << endl;
  }
  return 0;
}

