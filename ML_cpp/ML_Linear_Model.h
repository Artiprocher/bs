#ifndef ML_Linear_Model
#define ML_Linear_Model

#include <bits/stdc++.h>
using namespace std;
#include "ML_Rand.h"
#include "ML_Vector.h"

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
        cout << " y =";
        each_index(i, w) {
            cout << " "[i == 0] << "(" << w[i] << ") * x" << i << " "
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
        for(auto i:w)if(isinf(i)){
            cerr<<"Divergence!"<<endl;
            exit(1);
        }
    }
    void save(const char *file_name) const {
        ofstream fout;
        fout.open(file_name, ios::out);
        fout << "LinearRegression" << endl;
        for (auto i : w) {
            fout << fixed << setprecision(10) << i << endl;
        }
    }
    void load(const char *file_name) {
        ifstream fin;
        fin.open(file_name, ios::in);
        w.clear();
        string model_name;
        fin >> model_name;
        if (model_name != "LinearRegression") {
            cerr << "It is not a Linear Regression Model." << endl;
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
        cout << " y = sigmoid(";
        each_index(i, w) {
            cout << " "[i == 0] << "(" << w[i] << ") * x" << i << " "
                      << "+)"[i + 1 == (int)w.size()];
        }
        cout << endl;
    }
    double predict(const Vector &x) const {
        assert(x.size() == w.size());
        return sigmoid(Dot(x, w));
    }
    void train(const Vector &x, const double &y) {
        assert(x.size() == w.size());
        double s = Dot(x, w), y_ = sigmoid(s);
        w -= (eta * (y_ - y) * sigmoid_diff(s)) * x;
        for(auto i:w)if(isinf(i)){
            cerr<<"Divergence!"<<endl;
            exit(1);
        }
    }
    void save(const char *file_name) const {
        ofstream fout;
        fout.open(file_name, ios::out);
        fout << "LogitRegression" << endl;
        for (auto i : w) {
            fout << fixed << setprecision(10) << i << endl;
        }
    }
    void load(const char *file_name) {
        ifstream fin;
        fin.open(file_name, ios::in);
        w.clear();
        string model_name;
        fin >> model_name;
        if (model_name != "LogitRegression") {
            cerr << "It is not a Logit Regression Model." << endl;
        }
        double temp;
        while (fin >> temp) {
            w.emplace_back(temp);
        }
    }
};
#endif
