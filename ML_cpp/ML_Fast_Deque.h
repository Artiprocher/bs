#include <bits/stdc++.h>

template <typename type,int maxn> class Deque{
protected:
    type data[maxn];
    int l,r,n;
public:
    Deque<type,maxn> (){
        n=0;
        l=0;
        r=0;
    }
    void clear(){
        n=0;
        l=0;
        r=0;
    }
    int size()const{
        return n;
    }
    bool empty()const{
        return n==0;
    }
    void push_front(const type &a){
        assert(n<maxn);
        l--;
        if(l==-1)l=maxn-1;
        data[l]=a;
        n++;
    }
    void push_back(const type &a){
        assert(n<maxn);
        data[r]=a;
        r++;
        if(r==maxn)r=0;
        n++;
    }
    void pop_front(){
        assert(n>0);
        l++;
        if(l==maxn)l=0;
        n--;
    }
    void pop_back(){
        assert(n>0);
        r--;
        if(r==-1)r=maxn-1;
        n--;
    }
    type front()const{
        assert(n>0);
        return data[l];
    }
    type back()const{
        assert(n>0);
        return data[r==0?(maxn-1):(r-1)];
    }
    type &operator [] (int i){
        assert(i>=0 && i<n);
        return data[(l+i)>=maxn?(l+i-maxn):(l+i)];
    }
};