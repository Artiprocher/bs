#include <bits/stdc++.h>
#include "ML_Rand.h"
#include "ML_Vector.h"
#ifndef ML_Neural_Network
#define ML_Neural_Network

namespace OLD{
enum ActivationFunction { CONSTANT, SIGMOID, SIN, RELU };
std::function<double(double)> constant = [](double x) { return x; };
std::function<double(double)> constant_diff = [](double x) { return 1.0; };
std::function<double(double)> sigmoid = [](double x) {
    return 1.0 / (1.0 + exp(-x));
};
std::function<double(double)> sigmoid_diff = [](double x) {
    double temp = exp(-x);
    return temp / ((1.0 + temp) * (1.0 + temp));
};
std::function<double(double)> Sin = [](double x) { return std::sin(x); };
std::function<double(double)> Cos = [](double x) { return std::sin(x); };
std::function<double(double)> relu = [](double x) {return x>0?x:0.0;};
std::function<double(double)> relu_diff = [](double x) {return x>0?1.0:0.0;};

class Layer {
   public:
    std::function<double(double)> f, f_;
    std::vector<Vector> w;
    Vector val, diff_val, in_val, c;
    int flag;
    int size() const { return val.size(); }
    void resize(int size) { val = diff_val = in_val = c = Vector(size, 0); }
    void init(int activation_function_flag, int siz) {
        flag = activation_function_flag;
        resize(siz);
        if (activation_function_flag == CONSTANT) {
            f = constant;
            f_ = constant_diff;
        } else if (activation_function_flag == SIGMOID) {
            f = sigmoid;
            f_ = sigmoid_diff;
        } else if (activation_function_flag == SIN) {
            f = Sin;
            f_ = Cos;
        } else if (activation_function_flag == RELU){
            f = relu;
            f_ = relu_diff;
        } else {
            assert(false);
        }
    }
    void resetWeight() {
        for (auto &i : w) {
            for (auto &j : i) j = Rand(-0.5, 0.5);
        }
        for (auto &i : c) {
            i = Rand() - 0.5;
        }
    }
    void resetWeight(double l, double r) {
        for (auto &i : w) {
            for (auto &j : i) j = Rand(l, r);
        }
        for (auto &i : c) {
            i = Rand(l, r);
        }
    }
};
void connect(Layer &a, Layer &b) {
    b.w.resize(b.size());
    for (auto &i : b.w) i.resize(a.size());
}
class BP_Network {
   public:
    double eta = 0.1;
    std::vector<Layer> L;
    void show() const {
        for (int i = 0; i < (int)L.size(); i++) {
            std::cout << "Layer:" << i
                      << " activation_function_flag:" << L[i].flag << std::endl;
            std::cout << std::fixed << std::setprecision(3);
            int idx = 0;
            for (auto j : (L[i].w)) {
                for (auto k : j) std::cout << " " << k;
                std::cout << " " << L[i].c[idx++];
                std::cout << std::endl;
            }
        }
    }
    void init(const std::vector<int> &size, const std::vector<int> &flag) {
        assert(size.size() == flag.size());
        L.resize(size.size());
        for (int i = 0; i < (int)size.size(); i++) {
            L[i].init(flag[i], size[i]);
        }
        for (int i = 0; i + 1 < (int)size.size(); i++) {
            connect(L[i], L[i + 1]);
        }
        for (int i = 1; i < (int)size.size(); i++) {
            L[i].resetWeight();
        }
    }
    void push_forward(const Vector &x) {
        assert((int)x.size() == (int)L[0].size());
        for (int i = 0; i < L[0].size(); i++) {
            L[0].in_val[i] = x[i];
            L[0].val[i] = L[0].f(L[0].in_val[i]);
        }
        for (int i = 1; i < (int)L.size(); i++) {
            for (int j = 0; j < L[i].size(); j++) {
                L[i].in_val[j] = L[i].c[j];
                for (int k = 0; k < L[i - 1].size(); k++) {
                    L[i].in_val[j] += L[i - 1].val[k] * L[i].w[j][k];
                }
                L[i].val[j] = L[i].f(L[i].in_val[j]);
            }
        }
    }
    void push_backward(const Vector &y) {
        assert((int)y.size() == (int)L.back().size());
        for (int i = 0; i < L.back().size(); i++) {
            L.back().diff_val[i] = L.back().val[i] - y[i];
        }
        for (int i = (int)L.size() - 2; i >= 1; i--) {
            for (int j = 0; j < L[i].size(); j++) {
                L[i].diff_val[j] = 0;
                for (int k = 0; k < L[i + 1].size(); k++) {
                    L[i].diff_val[j] += L[i + 1].diff_val[k] *
                                        L[i + 1].f_(L[i + 1].in_val[k]) *
                                        L[i + 1].w[k][j];
                }
            }
        }
    }
    Vector predict(const Vector &x) {
        push_forward(x);
        return L.back().val;
    }
    void train(const Vector &x, const Vector &y) {
        push_forward(x);
        push_backward(y);
        double temp;
        for (int i = 1; i < (int)L.size(); i++) {
            for (int j = 0; j < L[i].size(); j++) {
                temp = eta * L[i].diff_val[j] * L[i].f_(L[i].in_val[j]);
                for (int k = 0; k < L[i - 1].size(); k++) {
                    L[i].w[j][k] -= temp * L[i - 1].val[k];
                    if (std::isinf(L[i].w[j][k])) {
                        std::cerr << "Divergence!" << std::endl;
                        exit(0);
                    }
                }
                L[i].c[j] -= temp;
            }
        }
    }
};
}
#endif