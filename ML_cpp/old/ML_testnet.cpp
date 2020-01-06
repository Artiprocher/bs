#include <bits/stdc++.h>
#define rep(i,a,b) for(int i=a;i<=b;i++)
#include "ML_Vector.h"
#include "ML_Rand.h"

enum ActivationFunction{CONSTANT,SIGMOID};
double constant(double x){
    return x;
}
double constant_diff(double x){
    return 1.0;
}
double sigmoid(double x){
    return 1.0/(1.0+exp(-x));
}
double sigmoid_diff(double x){
    double temp = exp(-x);
    return temp / ((1 + temp) * (1 + temp));
}

class Layer{
public:
    double (*f) (double x);
    double (*f_) (double x);
    std::vector<Vector> w,dw;
    Vector val,diff_val,in_val;
    int flag;
    int size()const{
        return val.size();
    }
    void resize(int size){
        val=diff_val=in_val=Vector(size,0);
    }
    void init(int activation_function_flag,int siz){
        flag=activation_function_flag;
        resize(siz);
        if(activation_function_flag==CONSTANT){
            f=constant;
            f_=constant_diff;
        }else if(activation_function_flag==SIGMOID){
            f=sigmoid;
            f_=sigmoid_diff;
        }else{
            assert(false);
        }
    }
    void resetWeight(){
        for(auto &i:w){
            for(auto &j:i)j=Rand()-0.5;
        }
    }
};
void connect(Layer &a,Layer &b){
    b.w.resize(b.size());
    for(auto &i:b.w)i.resize(a.size());
    b.dw=b.w;
}
class BP_Network{
public:
    double eta=0.1;
    std::vector<Layer> L;
    void show()const{
        for(int i=0;i<L.size();i++){
            std::cout<<"Layer:"<<i<<" activation_function_flag:"<<L[i].flag<<std::endl;
            std::cout<<std::fixed<<std::setprecision(3);
            for(auto j:(L[i].w)){
                for(auto i:j)std::cout<<" "<<i;
                std::cout<<std::endl;
            }
        }
    }
    void init(const std::vector<int> &size,const std::vector<int> &flag){
        assert(size.size()==flag.size());
        L.resize(size.size());
        for(int i=0;i<size.size();i++){
            L[i].init(flag[i],size[i]);
        }
        for(int i=0;i+1<size.size();i++){
            connect(L[i],L[i+1]);
        }
        for(int i=1;i<size.size();i++){
            L[i].resetWeight();
        }
    }
    void push_forward(const Vector &x){
        assert(x.size()==L[0].size());
        for(int i=0;i<L[0].size();i++){
            L[0].in_val[i]=L[0].val[i]=x[i];
        }
        for(int i=1;i<L.size();i++){
            for(int j=0;j<L[i].size();j++){
                L[i].in_val[j]=0;
                for(int k=0;k<L[i-1].size();k++){
                    L[i].in_val[j]+=L[i-1].val[k]*L[i].w[j][k];
                }
                L[i].val[j]=(*L[i].f)(L[i].in_val[j]);
            }
        }
    }
    void push_backward(const Vector &y){
        assert(y.size()==L.back().size());
        for(int i=0;i<L.back().size();i++){
            L.back().diff_val[i]=L.back().val[i]-y[i];
        }
        for(int i=(int)L.size()-2;i>=0;i--){
            for(int j=0;j<L[i].size();j++){
                L[i].diff_val[j]=0;
                for(int k=0;k<L[i+1].size();k++){
                    L[i].diff_val[j]+=L[i+1].diff_val[k]*L[i+1].f_(L[i+1].in_val[k])*L[i+1].w[k][j];
                }
            }
        }
    }
    Vector predict(const Vector &x){
        push_forward(x);
        return L.back().val;
    }
    void train(const Vector &x,const Vector &y){
        push_forward(x);
        push_backward(y);
        for(int i=1;i<L.size();i++){
            for(int j=0;j<L[i].size();j++){
                for(int k=0;k<L[i-1].size();k++){
                    L[i].dw[j][k]=L[i].diff_val[j]*L[i].f_(L[i].in_val[j])*L[i-1].val[k];
                    L[i].w[j][k]-=eta*L[i].dw[j][k];
                }
            }
        }
    }
};
using namespace std;
int main(){
    BP_Network net;
    net.init({2,2,2,2},{CONSTANT,CONSTANT,CONSTANT,CONSTANT});
    net.show();
    rep(it,1,10000){
        double a=Rand(),b=Rand();
        net.train({a,b},{a+b,a-b});
    }
    net.show();
    double a,b;
    while(cin>>a>>b){
        cout<<net.predict({a,b})<<endl;
    }
    return 0;
}
/*

*///
