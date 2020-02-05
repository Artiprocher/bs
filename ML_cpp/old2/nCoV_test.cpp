#include <bits/stdc++.h>
#include "ML_Model.h"
#define rep(i, a, b) for (int i = (a); i <= (int)(b); i++)
using namespace std;

/*神经网络计算图节点*/
class Node;
/*神经网络边*/
class Edge;
/*激活函数节点*/
class ActiveNode;
/*网络结构参数*/
class LayerProfile;

class LayerProfile {
   public:
    /*神经元数量*/
    int size;
    /*是否需要阈值 1:需要 0:不需要*/
    int threshold;
    /*激活函数与激活函数导数*/
    function<double(double)> f, f_;
    /*学习率*/
    double eta;
};
typedef vector<LayerProfile> NetworkProfile;

class Node {
   public:
    /*神经元的传入数据、传出数据、导数值*/
    double in_val, out_val, diff_val;
    /*学习率*/
    double eta;
    /*时间戳*/
    int time_flag;
    /*下标信息*/
    int index;
    /*是否需要阈值 1:需要 0:不需要*/
    int threshold;
    /*是否需要反向传值 1:需要 0:不需要*/
    int need_push_backward;
    /*以该神经元为终点和起点的边*/
    vector<Edge *> pre, nex;
    /*输出神经元相关信息*/
    virtual void show(ostream &o) const { o << "Empty Node" << endl; };
    /*正向传值到该神经元*/
    virtual void add_forward(double x){};
    /*反向传值到该神经元*/
    virtual void add_backward(double x){};
    /*从该神经元正向传值*/
    virtual void push_forward(){};
    /*从该神经元反向传值*/
    virtual void push_backward(){};
    /*更新权重*/
    virtual void update_w(){};
    /*设置激活函数*/
    virtual void set_active_function(function<double(double)> F,
                                     function<double(double)> F_){};
    /*取消激活函数*/
    virtual void threshold_cancel(){};
};

class Edge {
   public:
    double w;
    Node *u;
    Node *v;
    Edge() { w = Rand(0.0, 1.0); }
    void show(ostream &o) const {
        o << (u->index);
        o << " -> ";
        o << (v->index);
        o << " w=" << w;
        o << endl;
    }
};

class ActiveNode : public Node {
   public:
    double c;
    function<double(double)> f, f_;
    ActiveNode() {
        time_flag = Timer::timer;
        eta = 0.1;
        threshold = 1;
        need_push_backward = 1;
        c = Rand(0.0, 1.0);
    }
    virtual void show(ostream &o) const {
        o << "ActiveNode";
        o << " index=" << index;
        if (threshold == 1) o << " c=" << c;
        o << endl;
    }
    virtual void set_active_function(function<double(double)> F,
                                     function<double(double)> F_) {
        f = F;
        f_ = F_;
        threshold = 1;
    }
    virtual void threshold_cancel() {
        threshold = 0;
        c = 0;
    }
    virtual void add_forward(double x) {
        if (time_flag != Timer::timer) {
            time_flag = Timer::timer;
            in_val = c + x;
        } else {
            in_val += x;
        }
    }
    virtual void push_forward() {
        out_val = f(in_val);
        for (auto &e : nex) {
            e->v->add_forward(out_val * (e->w));
        }
    }
    virtual void add_backward(double x) {
        if (time_flag != Timer::timer) {
            time_flag = Timer::timer;
            diff_val = x;
        } else {
            diff_val += x;
        }
    }
    virtual void push_backward() {
        diff_val *= f_(in_val);
        if (need_push_backward) {
            for (auto &e : pre) {
                e->u->add_backward(diff_val * (e->w));
            }
        }
    }
    virtual void update_w() {
        for (auto &e : pre) {
            e->w -= eta * diff_val * (e->u->out_val);
        }
        if (threshold == 1) c -= eta * diff_val;
    }
};

Edge *connect(Node *u, Node *v) {
    Edge *e = new Edge;
    e->u = u;
    e->v = v;
    u->nex.emplace_back(e);
    v->pre.emplace_back(e);
    return e;
}

class Graph {
   public:
    /*所有神经元*/
    vector<Node *> L;
    /*输入层和输出层*/
    vector<Node *> X, Y;
    /*正向拓扑序与反向拓扑序*/
    vector<Node *> LA, LB;
    /*layer位置信息*/
    vector<pair<int, int> > ly;
    /*所有边*/
    vector<Edge *> E;
    void show(ostream &o) const {
        o << "Node=" << L.size() << endl;
        for (auto &i : L) i->show(o);
        o << "Edge=" << E.size() << endl;
        for (auto &i : E) i->show(o);
    }
    void clear() {
        L.clear();
        E.clear();
        X.clear();
        Y.clear();
        LA.clear();
        LB.clear();
        ly.clear();
    }
    void connect_layer(pair<int, int> a, pair<int, int> b) {
        for (int i = a.first; i < a.second; i++) {
            for (int j = b.first; j < b.second; j++) {
                E.emplace_back(connect(L[i], L[j]));
            }
        }
    }
    void init(const NetworkProfile &info) {
        clear();
        int num = 0;
        for (auto i : info) num += i.size;
        L.resize(num);
        int index = 0;
        for (auto i : info) {
            pair<int, int> p;
            p.first = index;
            for (int j = 0; j < i.size; j++) {
                L[index] = new ActiveNode;
                L[index]->index = index;
                L[index]->set_active_function(i.f, i.f_);
                L[index]->eta = i.eta;
                if (i.threshold == 0) L[index]->threshold_cancel();
                index++;
            }
            p.second = index;
            ly.emplace_back(p);
        }
        X = vector<Node *>(L.begin(), L.begin() + info[0].size);
        Y = vector<Node *>(L.end() - info.back().size, L.end());
        for (int i = 0; i + 1 < (int)ly.size(); i++) {
            connect_layer(ly[i], ly[i + 1]);
        }
        Timer::timer++;
    }
    void prepare() {
        static vector<int> in_deg;
        static Deque<Node *, 10000> q;
        in_deg.resize(L.size());
        //正向拓扑排序
        LA.clear();
        q.clear();
        for (auto &i : in_deg) i = 0;
        for (auto &i : E) in_deg[i->v->index]++;
        for (auto &i : L)
            if (in_deg[i->index] == 0) q.push_back(i);
        while (!q.empty()) {
            LA.emplace_back(q.front());
            for (auto &e : q.front()->nex) {
                in_deg[e->v->index]--;
                if (in_deg[e->v->index] == 0) {
                    q.push_back(e->v);
                }
            }
            q.pop_front();
        }
        assert(LA.size() == L.size());
        //反向拓扑排序
        LB.clear();
        q.clear();
        for (auto &i : in_deg) i = 0;
        for (auto &i : E) in_deg[i->u->index]++;
        for (auto &i : L)
            if (in_deg[i->index] == 0) q.push_back(i);
        while (!q.empty()) {
            LB.emplace_back(q.front());
            for (auto &e : q.front()->pre) {
                in_deg[e->u->index]--;
                if (in_deg[e->u->index] == 0) {
                    q.push_back(e->u);
                }
            }
            q.pop_front();
        }
        assert(LB.size() == L.size());
    }
    Vector predict(const Vector &x) {
        assert(x.size() == X.size());
        int index = 0;
        for (auto &i : X) i->add_forward(x[index++]);
        for (auto &i : LA) i->push_forward();
        Vector y(Y.size(), 0);
        for (int i = 0; i < (int)Y.size(); i++) {
            y[i] = Y[i]->out_val;
        }
        Timer::timer++;
        return y;
    }
    void train(const Vector &x, const Vector &y) {
        assert(x.size() == X.size() && y.size() == Y.size());
        predict(x);
        int index = 0;
        for (auto &i : Y) i->add_backward((i->out_val) - y[index++]);
        for (auto &i : LB) i->push_backward();
        for (auto &i : L) i->update_w();
        Timer::timer++;
    }
};

Graph net;
DataSet testx, testy;

function<double(double)> Exp=[](double x){return exp(x);};
function<double(double)> Exp_diff=[](double x){return exp(x);};

int main() {
    // read data
    testx={{{1},{2},{3},{4},{5},{6},{7},{8},{9}}};
    testy={{{2.91},{4.40},{5.71},{8.30},{12.87},{19.75},{27.44},{45.15},{59.74}}};
    // model init
    double eta=0.000005;
    NetworkProfile info = {{1, 0, constant, constant_diff, eta},
                           {1, 0, Exp, Exp_diff, eta},
                           {1, 1, constant, constant_diff, eta}};
    net.init(info);
    net.prepare();
    net.show(cout);
    // train
    int epoch = 100000000;
    rep(it, 1, epoch) {
        int idx = randint(0,8);
        net.train(testx.data[idx],testy.data[idx]);
    }
    net.show(cout);
    //test
    double x;
    while(cin>>x){
        cout<<net.predict({x})*100<<endl;
    }
    return 0;
}
/*

*///
