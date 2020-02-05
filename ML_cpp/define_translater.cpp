#include <bits/stdc++.h>
#define rep(i,a,b) for(ll i=a;i<=b;i++)
using namespace std;
typedef long long ll;

vector<string> data;

int main(){
    ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
    string s;
    while(getline(cin,s)){
        data.push_back(s);
    }
    int len=0;
    for(auto s:data){
        len=max(len,(int)s.size());
    }
    for(auto s:data){
        cout<<s;
        rep(i,1,len-s.size()+1)cout<<' ';
        cout<<'\\'<<endl;
    }
    return 0;
}
/*

*///
