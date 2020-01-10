#include <bits/stdc++.h>
#include "ML_Rand.h"
#include "ML_Vector.h"

#ifndef ML_Linear_Model
#define ML_Linear_Model

class LinearRegression {
   private:
    Vector w;

   public:
    double eta = 0.1;
    void init(int n) {
        w.resize(n);
        for (auto &i : w) i = Rand() - 0.5;
    }
    LinearRegression() {}
    LinearRegression(int n) { init(n); }
    void show() const {
        std::cout << " y =";
        each_index(i, w) {
            std::cout << " "[i == 0] << "(" << w[i] << ") * x" << i << " "
                      << "+\n"[i + 1 == (int)w.size()];
        }
    }
    double predict(const Vector &x) const {
        assert(x.size() == w.size());
        return Dot(x, w);
    }
    void train(const Vector &x, const double &y) {
        assert(x.size() == w.size());
        double y_ = predict(x);
        w -= (eta * (y_ - y)) * x;
        for(auto i:w)if(std::isinf(i)){
            std::cerr<<"Divergence!"<<std::endl;
            exit(1);
        }
    }
    void save(const char *file_name) const {
        std::ofstream fout;
        fout.open(file_name, std::ios::out);
        fout << "LinearRegression" << std::endl;
        for (auto i : w) {
            fout << std::fixed << std::setprecision(10) << i << std::endl;
        }
    }
    void load(const char *file_name) {
        std::ifstream fin;
        fin.open(file_name, std::ios::in);
        w.clear();
        std::string model_name;
        fin >> model_name;
        if (model_name != "LinearRegression") {
            std::cerr << "It is not a Linear Regression Model." << std::endl;
        }
        double temp;
        while (fin >> temp) {
            w.emplace_back(temp);
        }
    }
};

class LogitRegression {
   private:
    Vector w;

   public:
    double eta = 0.1;
    static double sigmoid(double x) { return 1.0 / (1 + exp(-x)); }
    static double sigmoid_diff(double x) {
        double temp = exp(-x);
        return temp / ((1 + temp) * (1 + temp));
    }
    void init(int n) {
        w.resize(n);
        for (auto &i : w) i = Rand() - 0.5;
    }
    LogitRegression() {}
    LogitRegression(int n) { init(n); }
    void show() const {
        std::cout << " y = sigmoid(";
        each_index(i, w) {
            std::cout << " "[i == 0] << "(" << w[i] << ") * x" << i << " "
                      << "+)"[i + 1 == (int)w.size()];
        }
        std::cout << std::endl;
    }
    double predict(const Vector &x) const {
        assert(x.size() == w.size());
        return sigmoid(Dot(x, w));
    }
    void train(const Vector &x, const double &y) {
        assert(x.size() == w.size());
        double s = Dot(x, w), y_ = sigmoid(s);
        w -= (eta * (y_ - y) * sigmoid_diff(s)) * x;
        for(auto i:w)if(std::isinf(i)){
            std::cerr<<"Divergence!"<<std::endl;
            exit(1);
        }
    }
    void save(const char *file_name) const {
        std::ofstream fout;
        fout.open(file_name, std::ios::out);
        fout << "LogitRegression" << std::endl;
        for (auto i : w) {
            fout << std::fixed << std::setprecision(10) << i << std::endl;
        }
    }
    void load(const char *file_name) {
        std::ifstream fin;
        fin.open(file_name, std::ios::in);
        w.clear();
        std::string model_name;
        fin >> model_name;
        if (model_name != "LogitRegression") {
            std::cerr << "It is not a Logit Regression Model." << std::endl;
        }
        double temp;
        while (fin >> temp) {
            w.emplace_back(temp);
        }
    }
};
#endif
