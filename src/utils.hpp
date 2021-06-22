#ifndef UTILS
#define UTILS

#include "graph.hpp"
#include <vector>
#include <fstream>

using namespace std;

DiGraph & read_graph(DiGraph &, string);
pair<int, int> vertex_edge_num(string);
int edge_num(string);
vector<DiGraph> partition(string, int, int);

#endif
