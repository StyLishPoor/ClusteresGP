#include <fstream>
#include <string>
#include "utils.hpp"

DiGraph & read_graph(DiGraph & g, string path)
{
  Vertex src, dst;
  string tmp;
  bool hoge = true;
  ifstream ifs(path);
  if (!ifs) {
    cerr << "Cannot open file" << endl;
    exit(1);
  }
  while (getline(ifs, tmp)) {
    if (!isdigit(tmp[0])) continue;
    src = stoi(tmp.substr(0, tmp.find('\t')));
    dst = stoi(tmp.substr(tmp.find('\t')));
    g.add_vertex(src);
    g.add_vertex(dst);
    g.add_edge(src, dst);
    g.edge_num++;

  }
  return g;
}

pair<int, int> vertex_edge_num(string path)
{
  ifstream ifs(path);
  if (!ifs) {
    cerr << "Cannot open file" << endl;
    exit(1);
  }
  pair<int, int> num; // vertex : edge
  string tmp;
  while (getline(ifs, tmp)) {
    if ((tmp.find("Nodes") != string::npos) && (tmp.find("Edges") != string::npos)) {
      num.first = stoi(tmp.substr(tmp.find("Nodes") + 7, tmp.find("Edges") - 1));
      num.second = stoi(tmp.substr(tmp.find("Edges") + 7));
      break;
    }
  }
  return num;
}

vector<DiGraph> partition(string path, int edge_num, int graph_num)
{
  ifstream ifs(path);
  if (!ifs) {
    cerr << "Cannot open file" << endl;
    exit(1);
  }
  int high_graph = edge_num % graph_num; // あまりを負担するグラフ
  int edge_graph = edge_num / graph_num;
  cout << edge_graph << endl;
  int graph_id = 0;
  vector<DiGraph> graphs(graph_num);
  int src, dst;
  int count = 0;
  string tmp;
  while (getline(ifs, tmp)) {
    if (!isdigit(tmp[0])) continue;
    if (graph_id < high_graph) {
      if (count < edge_graph + 1) {
        src = stoi(tmp.substr(0, tmp.find('\t')));
        dst = stoi(tmp.substr(tmp.find('\t')));
        graphs[graph_id].add_vertex(src);
        graphs[graph_id].add_vertex(dst);
        graphs[graph_id].add_edge(src, dst);
        count++;
        continue;
      } else {
        count = 0;
        graph_id++;
      }
    } else {
      if (count < edge_graph) {
        src = stoi(tmp.substr(0, tmp.find('\t')));
        dst = stoi(tmp.substr(tmp.find('\t')));
        graphs[graph_id].add_vertex(src);
        graphs[graph_id].add_vertex(dst);
        graphs[graph_id].add_edge(src, dst);
        count++;
        continue;
      } else {
        count = 0;
        graph_id++;
      }
    }
  }
  return graphs;
}

