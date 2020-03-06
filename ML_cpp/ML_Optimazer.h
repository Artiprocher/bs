#ifndef ML_Optimazer
#define ML_Optimazer

#include <bits/stdc++.h>
using namespace std;
//参数列表
class ParameterList{
public:
    vector<double*> w,dw;
    vector<bool*> frozen;
    void clear(){
        w.clear();
        dw.clear();
        frozen.clear();
    }
    void add_parameter(double &x,double &dx){
        w.emplace_back(&x);
        dw.emplace_back(&dx);
        frozen.emplace_back(new bool(false));
    }
    void add_parameter(double &x,double &dx,bool &frozen_flag){
        w.emplace_back(&x);
        dw.emplace_back(&dx);
        frozen.emplace_back(&frozen_flag);
    }
    void load(ifstream &f){
        for(auto &i:w)f>>(*i);
    }
    void save(ofstream &f){
        for(auto &i:w)f<<fixed<<setprecision(10)<<(*i)<<endl;
    }
};

//优化器
namespace Optimazer{
//梯度下降法
class GradientDescent{
public:
    double eta=0.05;
    GradientDescent(){}
    GradientDescent(double eta):eta(eta){}
    void init(const ParameterList &L){}
    void iterate(const ParameterList &L){
        int n=L.w.size();
        for(int i=0;i<n;i++)if(!(*L.frozen[i])){
            *(L.w[i])-=eta*(*(L.dw[i]));
        }
    }
};
class Adam{
public:
    double alpha=0.001,beta1=0.9,beta2=0.999,eps=1e-8;
    double power_beta1=1,power_beta2=1;
    vector<double> m,v,m_,v_;
    long long t=0;
    void init(const ParameterList &L){
        int n=L.w.size();
        m=v=m_=v_=vector<double>(n,0);
        t=0;
        power_beta1=1,power_beta2=1;
    }
    void load(ifstream &f){
        f>>t;
        for(auto &i:m)f>>i;
        for(auto &i:v)f>>i;
    }
    void save(ofstream &f){
        f<<t<<endl;
        for(auto &i:m)f<<fixed<<setprecision(10)<<i<<endl;
        for(auto &i:v)f<<fixed<<setprecision(10)<<i<<endl;
    }
    void iterate(const ParameterList &L){
        int n=L.w.size();
        t++;
#ifdef DEBUG
        for(int i=0;i<n;i++){
            assert(!isnan(*L.w[i]));
            assert(!isinf(*L.w[i]));
            assert(!isnan(*L.dw[i]));
            assert(!isinf(*L.dw[i]));
        }
#endif
        if(t<10000){
            power_beta1*=beta1,power_beta2*=beta2;
            for(int i=0;i<n;i++)if(!(*L.frozen[i])){
                m[i]=beta1*m[i]+(1.0-beta1)*(*L.dw[i]);
                v[i]=beta2*v[i]+(1.0-beta2)*(*L.dw[i])*(*L.dw[i]);
                m_[i]=m[i]/(1.0-power_beta1);
                v_[i]=v[i]/(1.0-power_beta2);
                (*L.w[i])-=alpha*m_[i]/(sqrt(v_[i])+eps);
            }
        }else{
            for(int i=0;i<n;i++)if(!(*L.frozen[i])){
                m[i]=beta1*m[i]+(1.0-beta1)*(*L.dw[i]);
                v[i]=beta2*v[i]+(1.0-beta2)*(*L.dw[i])*(*L.dw[i]);
                (*L.w[i])-=alpha*m[i]/(sqrt(v[i])+eps);
            }
        }
    }
};
}

#endif
