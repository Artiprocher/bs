#include <bits/stdc++.h>
#include "ML_Vector.h"
#include "ML_Rand.h"

#ifndef ML_Linear_Model
#define ML_Linear_Model

class LinearRegression {
private:
    Vector w;
public:
    double eta = 0.1;
    void init(int n) {
        w.resize(n);
        for (auto &i : w)i = Rand() - 0.5;
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
        for (auto &i : w)i = Rand() - 0.5;
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
