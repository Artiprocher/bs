#include <bits/stdc++.h>
#ifndef ML_Rand
#define ML_Rand
double Rand(){
    static std::default_random_engine generator(time(0));
    static std::uniform_real_distribution<double> rd(0.0,1.0);
    return rd(generator);
}
double Rand(double l,double r){
    static std::default_random_engine generator(time(0));
    static std::uniform_real_distribution<double> rd(0.0,1.0);
    return rd(generator)*(r-l)+l;
}
int randint(int l,int r){
    static std::mt19937 rd(time(0));
    return rd()%(r-l+1)+l;
}
#endif
