#include <bits/stdc++.h>
#include "ML_Model.h"
#define rep(i,a,b) for(int i=(a);i<=(int)(b);i++)
using namespace std;

class LayerProfile{
public:
    int size;
    int threshold;
    function<double(double)> f,f_;
};
typedef vector<LayerProfile> NetworkProfile;

class Node;/*神经网络计算图节点*/
class Edge;/*神经网络边*/
class ActiveNode;/*激活函数节点*/

class Node{
public:
    double in_val,out_val,diff_val;
    double eta;
    int time_flag,index;
    int threshold;
    vector<Edge*> pre,nex;
    virtual void show(ostream &o)const{
        o<<"Empty Node"<<endl;
    };
    virtual void add_forward(double x){};
    virtual void add_backward(double x){};
    virtual void push_forward(){};
    virtual void push_backward(){};
    virtual void update_w(){};
    virtual void set_active_function(function<double(double)> F,function<double(double)> F_){};
    virtual void threshold_cancel(){};
};
class Edge{
public:
    double w;
    Node *u;
    Node *v;
    Edge(){
        w=Rand(-0.5,0.5);
    }
    void show(ostream &o)const{
        o<<(u->index);
        o<<" -> ";
        o<<(v->index);
        o<<" w="<<w;
        o<<endl;
    }
};
class ActiveNode:public Node{
public:
    double c;
    function<double(double)> f,f_;
    ActiveNode(){
        time_flag=Timer::timer;
        eta=0.1;
        threshold=1;
        c=Rand(-0.5,0.5);
    }
    virtual void show(ostream &o)const{
        o<<"ActiveNode";
        o<<" index="<<index;
        if(threshold==1)o<<" c="<<c;
        o<<endl;
    }
    virtual void set_active_function(function<double(double)> F,function<double(double)> F_){
        f=F;
        f_=F_;
    }
    virtual void threshold_cancel(){
        threshold=0;
        c=0;
    }
    virtual void add_forward(double x){
        if(time_flag!=Timer::timer){
            time_flag=Timer::timer;
            in_val=c+x;
        }else{
            in_val+=x;
        }
    }
    virtual void push_forward(){
        out_val=f(in_val);
        for(auto &e:nex){
            e->v->add_forward(out_val*(e->w));
        }
    }
    virtual void add_backward(double x){
        if(time_flag!=Timer::timer){
            time_flag=Timer::timer;
            diff_val=x;
        }else{
            diff_val+=x;
        }
    }
    virtual void push_backward(){
        diff_val*=f_(in_val);
        for(auto &e:pre){
            e->u->add_backward(diff_val*(e->w));
        }
    }
    virtual void update_w(){
        for(auto &e:pre){
            e->w-=eta*diff_val*(e->u->out_val);
        }
        if(threshold==1)c-=eta*diff_val;
    }
};
Edge* connect(Node *u,Node *v){
    Edge *e=new Edge;
    e->u=u;
    e->v=v;
    u->nex.emplace_back(e);
    v->pre.emplace_back(e);
    return e;
}

class Graph{
public:
    vector<Node*> L,X,Y,LA,LB;
    vector<Edge*> E;
    void show(ostream &o)const{
        o<<"Node="<<L.size()<<endl;
        for(auto &i:L)i->show(o);
        o<<"Edge="<<E.size()<<endl;
        for(auto &i:E)i->show(o);
    }
    void clear(){
        L.clear();
        E.clear();
        X.clear();
        Y.clear();
        LA.clear();
        LB.clear();
    }
    void init(const vector<int> &size,const vector< function<double(double)> > &f,const vector< function<double(double)> > &f_){
        clear();
        assert(size.size()==f.size() && f.size()==f_.size());
        L.resize(accumulate(size.begin(),size.end(),0));
        int index=0;
        for(int i=0;i<(int)size.size();i++){
            for(int j=0;j<size[i];j++){
                L[index]=new ActiveNode;
                L[index]->index=index;
                L[index]->set_active_function(f[i],f_[i]);
                index++;
            }
        }
        X=vector<Node*>(L.begin(),L.begin()+size[0]);
        Y=vector<Node*>(L.end()-size.back(),L.end());
        index=0;
        for(int i=0;i<(int)size.size()-1;i++){
            for(int j=0;j<size[i];j++){
                for(int k=0;k<size[i+1];k++){
                    E.emplace_back(connect(L[index+j],L[index+size[i]+k]));
                }
            }
            index+=size[i];
        }
        for(auto &i:X)i->threshold_cancel();
        Timer::timer++;
    }
    void set_eta(double e){
        for(auto &i:L)i->eta=e;
    }
    void prepare(){
        static vector<int> in_deg;
        static Deque<Node*,10000> q;
        in_deg.resize(L.size());
        //正向拓扑排序
        LA.clear();
        q.clear();
        for(auto &i:in_deg)i=0;
        for(auto &i:E)in_deg[i->v->index]++;
        for(auto &i:L)if(in_deg[i->index]==0)q.push_back(i);
        while(!q.empty()){
            LA.emplace_back(q.front());
            for(auto &e:q.front()->nex){
                in_deg[e->v->index]--;
                if(in_deg[e->v->index]==0){
                    q.push_back(e->v);
                }
            }
            q.pop_front();
        }
        assert(LA.size()==L.size());
        //反向拓扑排序
        LB.clear();
        q.clear();
        for(auto &i:in_deg)i=0;
        for(auto &i:E)in_deg[i->u->index]++;
        for(auto &i:L)if(in_deg[i->index]==0)q.push_back(i);
        while(!q.empty()){
            LB.emplace_back(q.front());
            for(auto &e:q.front()->pre){
                in_deg[e->u->index]--;
                if(in_deg[e->u->index]==0){
                    q.push_back(e->u);
                }
            }
            q.pop_front();
        }
        assert(LB.size()==L.size());
    }
    Vector predict(const Vector &x){
        assert(x.size()==X.size());
        int index=0;
        for(auto &i:X)i->add_forward(x[index++]);
        for(auto &i:LA)i->push_forward();
        Vector y(Y.size(),0);
        for(int i=0;i<(int)Y.size();i++){
            y[i]=Y[i]->out_val;
        }
        Timer::timer++;
        return y;
    }
    void train(const Vector &x,const Vector &y){
        assert(x.size()==X.size() && y.size()==Y.size());
        predict(x);
        int index=0;
        for(auto &i:Y)i->add_backward((i->out_val)-y[index++]);
        for(auto &i:LB)i->push_backward();
        for(auto &i:L)i->update_w();
        Timer::timer++;
    }
};

CSV_Reader csv_reader;
Graph net;
DataSet trainx, trainy, testx, testy;

void show_image(const Vector &a) {
    rep(i, 0, 783) {
        cout << (a[i] > 0.5 ? "*" : " ");
        if ((i + 1) % 28 == 0) cout << endl;
    }
    cout << endl;
}
void judge(const DataSet &testx, const DataSet &testy) {
    int all = testx.data.size(), ac = 0;
    rep(it, 0, all - 1) {
        Vector a = net.predict(testx.data[it]);
        int ans = 0;
        rep(i, 1, 9) {
            if (a[i] > a[ans]) ans = i;
        }
        if (testy.data[it][ans] > 0.5) ac++;
    }
    cout << (ac * 1.0 / all) << endl;
}

int main(){
    NetworkProfile info={
        {784,0,constant,constant_diff},
        {10,1,sigmoid,sigmoid_diff},
        {10,1,sigmoid,sigmoid_diff}
    };
    // read data
    cout << "Reading data" << endl;
    csv_reader.open("train.csv");
    //csv_reader.shuffle();
    int split_position = 30000;
    csv_reader.export_number_data(1, split_position, 1, 784, trainx);
    csv_reader.export_onehot_data(1, split_position, 0, trainy);
    csv_reader.export_number_data(split_position + 1, 42000, 1, 784, testx);
    csv_reader.export_onehot_data(split_position + 1, 42000, 0, testy);
    csv_reader.close();
    rep(i, 0, trainx.data.size() - 1) trainx.data[i] *= 1.0 / 255;
    rep(i, 0, testx.data.size() - 1) testx.data[i] *= 1.0 / 255;
    // model init
    net.init({784, 10, 10}, {constant, sigmoid, sigmoid}, {constant_diff, sigmoid_diff, sigmoid_diff});
    net.set_eta(0.5);
    net.prepare();
    net.show(cout);
    // train
    cout << "Training model" << endl;
    judge(testx, testy);
    int epoch = 1000000, goal = 1;
    rep(it, 1, epoch) {
        int idx = randint(0, split_position - 1);
        net.train(trainx.data[idx], trainy.data[idx]);
        if (it * 100 >= epoch * goal) {
            cout << it * 100.0 / epoch << "%" << endl;
            if (goal % 10 == 0) {
                cout << "accuracy:";
                judge(testx, testy);
            }
            goal++;
        }
    }
    return 0;
}
/*

*///
