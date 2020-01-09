#include <bits/stdc++.h>
#include "ML_Model.h"
#define rep(i,a,b) for(int i=a;i<=b;i++)
using namespace std;

namespace Timer{
int timer=0;
void reset(){
    timer=0;
}
}

class Node;/*神经网络计算图节点*/
class Edge;/*神经网络边*/
class LinearNode;/*线性组合节点*/
class ActiveNode;/*激活函数节点*/

class Edge{
public:
    double w;
    Node *u;
    Node *v;
    Edge(){
        w=Rand(-0.5,0.5);
    }
};
class Node{
public:
    double in_val,out_val,diff_val;
    double eta;
    int time_flag,index,in_deg;
    vector<Edge*> pre,nex;
    virtual void show(){};
    virtual void add_forward(double x){};
    virtual void add_backward(double x){};
    virtual void push_forward(){};
    virtual void push_backward(){};
    virtual void update_w(){};
    virtual void set_active_function(function<double(double)> F,function<double(double)> F_){};
};
class ActiveNode:public Node{
public:
    double c;
    function<double(double)> f,f_;
    ActiveNode(){
        time_flag=Timer::timer;
        eta=0.1;
        c=Rand(-0.5,0.5);
    }
    virtual void show(){
        cout<<"ActiveNode";
        cout<<" index="<<index;
        cout<<" c="<<c;
        cout<<endl;
    }
    virtual void set_active_function(function<double(double)> F,function<double(double)> F_){
        f=F;
        f_=F_;
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
        c-=eta*diff_val;
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
    vector<Node*> L,X,Y;
    vector<Edge*> E;
    void show(){
        cout<<"Node:"<<endl;
        for(auto &i:L){
            i->show();
        }
        cout<<"Edge:"<<endl;
        for(auto &i:E){
            cout<<(i->u->index)<<" -> "<<(i->v->index)<<" w="<<(i->w)<<endl;
        }
    }
    void clear(){
        for(auto &i:L)delete i;
        L.clear();
        for(auto &i:E)delete i;
        E.clear();
    }
    void init(const vector<int> &size,const vector< function<double(double)> > &f,const vector< function<double(double)> > &f_){
        L.resize(accumulate(size.begin(),size.end(),0));
        int index=0;
        for(int i=0;i<size.size();i++){
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
        Timer::timer++;
    }
    void set_eta(double e){
        for(auto &i:L)i->eta=e;
    }
    Vector predict(const Vector &x){
        assert(x.size()==X.size());
        int index=0;
        for(auto &i:X)i->add_forward(x[index++]);
        for(auto &i:L)i->in_deg=0;
        for(auto &i:E)i->v->in_deg++;
        static queue<Node*> q;
        for(auto &i:L)if(i->in_deg==0)q.push(i);
        while(!q.empty()){
            q.front()->push_forward();
            for(auto &e:q.front()->nex){
                e->v->in_deg--;
                if(e->v->in_deg==0){
                    q.push(e->v);
                }
            }
            q.pop();
        }
        Vector y(Y.size(),0);
        for(int i=0;i<Y.size();i++){
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
        for(auto &i:L)i->in_deg=0;
        for(auto &i:E)i->u->in_deg++;
        static queue<Node*> q;
        for(auto &i:L)if(i->in_deg==0)q.push(i);
        while(!q.empty()){
            q.front()->push_backward();
            for(auto &e:q.front()->pre){
                e->u->in_deg--;
                if(e->u->in_deg==0){
                    q.push(e->u);
                }
            }
            q.pop();
        }
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
    // read data
    cout << "Reading data" << endl;
    csv_reader.open("train.csv");
    csv_reader.shuffle();
    int split_position = 30000;
    csv_reader.export_number_data(1, split_position, 1, 784, trainx);
    csv_reader.export_onehot_data(1, split_position, 0, trainy);
    csv_reader.export_number_data(split_position + 1, 42000, 1, 784, testx);
    csv_reader.export_onehot_data(split_position + 1, 42000, 0, testy);
    csv_reader.close();
    rep(i, 0, trainx.data.size() - 1) trainx.data[i] *= 1.0 / 255;
    rep(i, 0, testx.data.size() - 1) testx.data[i] *= 1.0 / 255;
    // model init
    net.init({784, 60, 10}, {constant, sigmoid, sigmoid}, {constant_diff, sigmoid_diff, sigmoid_diff});
    net.set_eta(0.1);
    // train
    cout << "Training model" << endl;
    int epoch = 1000000;
    rep(it, 1, epoch) {
        int idx = randint(0, split_position - 1);
        net.train(trainx.data[idx], trainy.data[idx]);
        if (it % 10000 == 0) cout << it / 10000 << "%" << endl;
    }
    // judge
    cout << "Judging model" << endl;
    judge(testx, testy);
    return 0;
}
/*

*///
