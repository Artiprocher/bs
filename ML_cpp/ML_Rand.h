#ifndef ML_Rand
#define ML_Rand

#include <bits/stdc++.h>
using namespace std;

#ifdef DEBUG
double Rand(double l=0,double r=1){
    static default_random_engine generator(0);
    static uniform_real_distribution<double> rd(0.0,1.0);
    return rd(generator)*(r-l)+l;
}
int randint(int l,int r){
    static mt19937 rd(0);
    return rd()%(r-l+1)+l;
}
#else
double Rand(double l=0,double r=1){
    static default_random_engine generator(time(0));
    static uniform_real_distribution<double> rd(0.0,1.0);
    return rd(generator)*(r-l)+l;
}
int randint(int l,int r){
    static mt19937 rd(time(0));
    return rd()%(r-l+1)+l;
}
#endif

#endif
