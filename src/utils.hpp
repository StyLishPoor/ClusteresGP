#ifndef UTILS
#define UTILS

#include "graph.hpp"
#include <vector>
#include <fstream>

using namespace std;

DiGraph & read_graph(DiGraph &, ifstream &);
pair<int, int> vertex_edge_num(ifstream &);
int edge_num(ifstream &);
vector<DiGraph> partition(ifstream &, int, int);

#endif
