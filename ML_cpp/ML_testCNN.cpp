#include <bits/stdc++.h>
#include "ML_Model.h"
#define rep(i, a, b) for (int i = (a); i <= (int)(b); i++)
using namespace std;

class Layer{
public:
    ;
};

template <const int N> class ActiveLayer:public Layer{
public:
    double in_val[N],out_val[N],diff_val[N];
    void show(){
        cout<<" I am a ActiveLayer."<<endl;
        in_val[0]=123;
    }
};

template <const int N,const int H,const int W> class ImageLayer{
public:
    double pix[N][H][W];
    void show(){
        cout<<" I am a ImageLayer."<<endl;
    }
};

template <const int N,const int H,const int W> class ConvLayer:public ImageLayer<N,H,W>{
public:
    void show(){
        cout<<" I am a ConvLayer."<<endl;
    }
};

template <const int N1,const int N2,const int H2,const int W2>
void solve1(ActiveLayer<N1> &x,ConvLayer<N2,H2,W2> &y){
    x.show();
    y.show();
}

template <const int N1,const int N2,const int H2,const int W2>
void solve1(ActiveLayer<N1> &x,ImageLayer<N2,H2,W2> &y){
    x.show();
    y.show();
}

int main() {
    ActiveLayer<5> A;
    ConvLayer<3,10,10> B;
    ImageLayer<3,10,10> C;
    solve1(A,C);
    return 0;
}
/*

*///
