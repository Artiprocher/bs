#include <bits/stdc++.h>
#include "ML_Model.h"
#define rep(i,a,b) for(int i=a;i<=b;i++)
using namespace std;
typedef long long ll;

BP_Network net;

int main(){
    net.init({1,2,1},{CONSTANT,CONSTANT,SIN});
    net.eta=0.0001;
    for(auto &l:net.L)l.resetWeight(-10,10);
    net.show();
    double a,b;
    rep(it,1,1000000){
        int x=randint(1,100);
        a=x;
        b=(x%2==1)?1:-1;
        net.train({a},{b});
    }
    net.show();
    while(cin>>a){
        cout<<net.predict({a})<<endl;
    }
    return 0;
}
/*

*///
