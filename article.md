# 神经网络在数据挖掘中的应用

## 摘要

还没写

**关键词**：神经网络 数据挖掘 深度学习

## 线性模型

### 多元线性回归

#### 多元线性回归模型

假设有训练数据集 $\{x_i,y_i\}_{i=1}^m$ ，其中随机变量 $x_i\in \mathbb{R}^n,y_i\in \mathbb{R},i=1,2,\dots,m$ ，以及预测数据集 $\{x_i'\}_{i=1}^m$ ，我们需要得到预测数据集中每一个数据对应的 $y_i'$ 

在多元线性回归模型中， 假设 $x=(x_1,x_2,\dots,x_n)\in \mathbb{R}^n$ 与 $y\in \mathbb{R}$ 有关系
$$
y=\sum_{i=1}^nw_ix_i
$$
模型的训练过程就是根据 $x,y$ 求 $w$ ，预测过程就是根据 $w,x$ 求 $y$ 

#### 参数的确定

在训练过程中，我们要确定 $w$ ，使得误差的平方和（简称均方误差）最小，即
$$
\min\sum_{i=1}^m\left(\sum_{j=1}^nw_jx_{ij}-y_i\right)^2
$$
使用数学方法可以求出最优的 $w$ ，但是在实际应用中，由于数据量较大，很难用数学结论求得精确地最优解，因此可以使用梯度下降法得到 $w$ 的近似最优解

令目标函数
$$
E=\left(\sum_{j=1}^nw_jx_{ij}-y_i\right)^2
$$
计算 $E$ 对 $w$ 的梯度
$$
\frac{\partial E}{\partial w}=\left(\frac{\partial E}{\partial w_1},\frac{\partial E}{\partial w_2},\dots,\frac{\partial E}{\partial w_n}\right)=(2Ex_{i1},2Ex_{i2},\dots,2Ex_{in})
$$
梯度是函数变化最快的方向，向梯度的反方向移动 $w$ ，将会逐渐逼近最优解，即先随机确定 $w^{[0]}$ ，再以下面的公式进行迭代
$$
w^{[n+1]}=w^{[n]}-\eta\frac{\partial E}{\partial w^{[n]}}
$$
其中 $\eta$ 是常参数，称为学习率，具体数值根据实际问题调整

### 对数几率回归

#### 对数几率回归模型

线性回归模型原理简单，但是不够灵活，难以解决较为复杂的问题，在此基础上，有对数几率回归（又称逻辑回归），来让线性模型能够更好地解决分类问题

假设有训练数据集 $\{x_i,y_i\}_{i=1}^m$ ，其中随机变量 $x_i\in \mathbb{R}^n,y_i\in \{1,0\},i=1,2,\dots,m$ （ $y_i$ 表示数据的类别标记），以及预测数据集 $\{x_i'\}_{i=1}^m$ ，我们需要得到预测数据集中每一个数据对应的 $y_i'$ 

我们可以对数据属于两类的条件概率的比值进行预测，假设
$$
\ln \frac{P(y=1|x)}{P(y=0|x)}=\sum_{i=1}^nw_ix_i
$$
同时显然有
$$
P(y=1|x)+P(y=0|x)=1
$$
解得
$$
P(y=1|x)=\frac{e^{wx^T}}{1+e^{wx^T}}
$$

$$
P(y=0|x)=\frac{1}{1+e^{wx^T}}
$$

这样就能得到数据属于不同类别的概率

#### 参数的确定

与前面类似，需要构造目标函数来作为衡量 $w$ 的标准，与前面不同，不用均方误差，而是用似然概率的对数（这种方法又叫极大似然法），即
$$
\min\sum_{i=1}^m\ln P(y_i|x_i,w)
$$
取对数而不是原数值是为了让概率的乘法变成对数的加法，让计算变得简单

令
$$
E=\ln P(y_i|x_i,w)
$$
以 $y_i=0$ 为例，有
$$
\frac{\partial E}{\partial w_j}=-\frac{e^{wx^T}x_j}{1+e^{wx^T}}
$$
用同样的方法迭代即可得到 $w$ 


## 线性模型测试

本文将使用 C++ 语言搭建简单的神经网络

声明向量类
```cpp
#include <bits/stdc++.h>
#ifndef ML_Vector
#define ML_Vector
typedef std::vector<double> Vector;
#define each_index(i,a) for(int i=0;i<a.size();i++)
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
```

线性模型
```cpp
#include <bits/stdc++.h>
#ifndef ML_Linear_Model
#define ML_Linear_Model
double Rand() {return rand() * 1.0 / 32768;}

class LinearRegression {
private:
    Vector w;
public:
    double eta = 0.1;
    void init(int n) {
        w.resize(n);
        for (auto &i : w)i = Rand();
    }
    LinearRegression() {}
    LinearRegression(int n) {init(n);}
    void show()const {
        std::cout << " y =";
        each_index(i, w) {
            std::cout << " "[i == 0] << "(" << w[i] << ") * x"
                      << i << " " << "+\n"[i + 1 == w.size()];
        }
    }
    double predict(const Vector &x)const {
        assert(x.size() == w.size());
        return Dot(x, w);
    }
    void train(const Vector &x, const double &y) {
        assert(x.size() == w.size());
        double y_ = predict(x);
        w -= (eta * (y_ - y)) * x;
    }
};

class LogitRegression {
private:
    Vector w;
public:
    double eta = 0.1;
    static double sigmoid(double x) {
        return 1.0 / (1 + exp(-x));
    }
    static double sigmoid_diff(double x) {
        double temp = exp(-x);
        return temp / ((1 + temp) * (1 + temp));
    }
    void init(int n) {
        w.resize(n);
        for (auto &i : w)i = Rand();
    }
    LogitRegression() {}
    LogitRegression(int n) {init(n);}
    void show()const {
        std::cout << " y = sigmoid(";
        each_index(i, w) {
            std::cout << " "[i == 0] << "(" << w[i] << ") * x"
                      << i << " " << "+)"[i + 1 == w.size()];
        }
        std::cout << std::endl;
    }
    double predict(const Vector &x)const {
        assert(x.size() == w.size());
        return sigmoid(Dot(x, w));
    }
    void train(const Vector &x, const double &y) {
        assert(x.size() == w.size());
        double s = Dot(x, w), y_ = sigmoid(s);
        w -= (eta * (y_ - y) * sigmoid_diff(s)) * x;
    }
};
#endif
```


