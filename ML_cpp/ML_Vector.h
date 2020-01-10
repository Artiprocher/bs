#include <bits/stdc++.h>
#ifndef ML_Vector
#define ML_Vector
typedef std::vector<double> Vector;
#define each_index(i,a) for(int i=0;i<(int)a.size();i++)
Vector operator += (Vector &a,const Vector &b){each_index(i,a)a[i]+=b[i];return a;}
Vector operator -= (Vector &a,const Vector &b){each_index(i,a)a[i]-=b[i];return a;}
Vector operator *= (Vector &a,const Vector &b){each_index(i,a)a[i]*=b[i];return a;}
Vector operator /= (Vector &a,const Vector &b){each_index(i,a)a[i]/=b[i];return a;}
Vector operator + (const Vector &a,const Vector &b){Vector c=a;return c+=b;}
Vector operator - (const Vector &a,const Vector &b){Vector c=a;return c-=b;}
Vector operator * (const Vector &a,const Vector &b){Vector c=a;return c*=b;}
Vector operator / (const Vector &a,const Vector &b){Vector c=a;return c/=b;}
Vector operator += (Vector &a,const double &b){each_index(i,a)a[i]+=b;return a;}
Vector operator -= (Vector &a,const double &b){each_index(i,a)a[i]-=b;return a;}
Vector operator *= (Vector &a,const double &b){each_index(i,a)a[i]*=b;return a;}
Vector operator /= (Vector &a,const double &b){each_index(i,a)a[i]/=b;return a;}
Vector operator + (const Vector &a,const double b){Vector c=a;return c+=b;}
Vector operator - (const Vector &a,const double b){Vector c=a;return c-=b;}
Vector operator * (const Vector &a,const double b){Vector c=a;return c*=b;}
Vector operator / (const Vector &a,const double b){Vector c=a;return c/=b;}
Vector operator + (const double b,const Vector &a){Vector c=a;return c+=b;}
Vector operator - (const double b,const Vector &a){return Vector(a.size(),b)-a;}
Vector operator * (const double b,const Vector &a){Vector c=a;return c*=b;}
Vector operator / (const double b,const Vector &a){return Vector(a.size(),b)/a;}
Vector operator - (const Vector &a){Vector c=a;each_index(i,c)c[i]=-c[i];return c;}
double Dot(const Vector &a,const Vector &b){
    assert(a.size()==b.size());
    double ans=0;
    each_index(i,a)ans+=a[i]*b[i];
    return ans;
}
double sum(const Vector &a){return std::accumulate(a.begin(),a.end(),0.0);}
std::istream &operator >>(std::istream &in,Vector &a){double x;in>>x;a.push_back(x);return in;}
std::ostream &operator <<(std::ostream &out,const Vector &a){
    out<<'(';
    each_index(i,a){
        out<<a[i];
        if(i+1!=(int)a.size())out<<',';
    }
    out<<')';
    return out;
}
template <class fun> void each(Vector &a,fun op){
    each_index(i,a)op(a[i]);
}
#endif
