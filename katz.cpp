#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
using namespace std;

class Graph
{
    struct Vertex
    {
        unsigned int label;
        list<unsigned int> outwards;

        Vertex(unsigned int label) : label(label) {}
    };

    unordered_map<unsigned int, Vertex *> vertices;
public:
    ~Graph()
    {
        for (const auto & p : vertices)
            delete p.second;
    }

    void add_edges(unsigned int src, const list<unsigned int> & dests)
    {
        Vertex * s;
        if (vertices.count(src))
            s = vertices[src];
        else
            vertices[src] = s = new Vertex(src);
        for (unsigned int dest : dests)
        {
            if (!vertices.count(dest))
                vertices[dest] = new Vertex(dest);
            s->outwards.push_back(dest);
        }
    }

    template<typename Func>
    void all_path_lengths_action(
        unsigned int src,
        unsigned int dest,
        unsigned int depth,
        Func action,
        unsigned int thres = 5
    )
    {
        if (src == dest)
        {
            action(depth);
            return;
        }

        if (depth == thres)
            return;

        depth += 1;
        Vertex * s = vertices[src];
        for (unsigned int n : s->outwards)
            all_path_lengths_action(n, dest, depth, action);
    }

    bool has_node(unsigned int label)
    {
        return vertices.count(label);
    }

    double katz(unsigned int src, unsigned int dest, double beta = 0.5)
    {
        map<unsigned int, unsigned int> counter;
        all_path_lengths_action(
            src,
            dest,
            0,
            [&counter](unsigned int l) -> void
            {
                counter[l] += 1;
            }
        );

        auto factor = 1.0, sum = 0.0;
        unsigned ll = 0;
        for(const auto & p : counter)
        {
            while (ll < p.first)
            {
                factor *= beta;
                ++ll;
            }

            sum += factor * p.second;
        }

        return sum;
    }

    unsigned int node_count()
    {
        return vertices.size();
    }
};

int main()
{
    ifstream train_f("train.txt");
    string line;
    Graph g;
    while(getline(train_f, line))
    {
        stringstream ss(line);
        unsigned int src;
        ss >> src;
        list<unsigned int> dests;
        unsigned int dest;
        while(ss >> dest)
            dests.push_back(dest);
        g.add_edges(src, dests);
    }
    cout << "Number of nodes: " << g.node_count() << endl
         << "--Read Finish--" << endl;

    ifstream test_f("test-public.txt");
    ofstream output_f("output-cpp.csv");
    getline(test_f, line);
    output_f << "Id,Prediction" << endl;
    map<unsigned int, double> scores;
    while(getline(test_f, line))
    {
        stringstream ss(line);
        unsigned int id, src, dest;
        ss >> id >> src >> dest;
        double k;
        if (g.has_node(src) && g.has_node(dest))
            k = g.katz(src, dest);
        else
            k = 0;
        cout << "Current: " << id << " " << src << " " << dest << endl
             << "Katz: " << k << endl;
        scores[id] = k;
    }
    cout << "--Process Finish--" << endl;
    return 0;
}
