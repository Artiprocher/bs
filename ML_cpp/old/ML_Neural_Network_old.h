#include <bits/stdc++.h>
#include "ML_Tensor.h"
#include "ML_Vector.h"
#include "ML_Rand.h"
#ifndef ML_Neural_Network
#define ML_Neural_Network
double active_function(int flag, double x) {
    if (flag == 0) {
        return x;
    } else if (flag == 1) {
        return 1.0 / (1 + exp(-x));
    } else {
        assert(false);
    }
}
double active_function_diff(int flag, double x) {
    if (flag == 0) {
        return 1.0;
    } else if (flag == 1) {
        double temp = exp(-x);
        return temp / ((1 + temp) * (1 + temp));
    } else {
        assert(false);
    }
}
class NeuralNetworkLayer {
public:
    std::vector<Vector> w;
    Vector value, activeValue, diffValue;
    int activeFlag;
    int size()const {
        return value.size();
    }
    void show()const {
        for (auto i : w)std::cout << "  " << i << std::endl;
    }
    void set_input_layer(int siz) {
        activeFlag = 0;
        value.resize(siz);
        activeValue.resize(siz);
    }
    void set_connect_layer(const NeuralNetworkLayer &ly, int siz, int flag) {
        activeFlag = flag;
        w.resize(siz);
        for (auto &i : w)i.resize(ly.size());
        value.resize(siz);
        activeValue.resize(siz);
        diffValue.resize(siz);
    }
    void weight_initialize() {
        for (auto &i : w) {
            for (auto &j : i)j = Rand();
        }
    }
    void calculate(const Vector &pre) {
        for (int i = 0; i < w.size(); i++) {
            value[i] = Dot(pre, w[i]);
            activeValue[i] = active_function(activeFlag, value[i]);
        }
    }
};
class NeuralNetwork {
private:
    Tensor dw;
public:
    std::vector<NeuralNetworkLayer> ly;
    double eta = 0.1;
    NeuralNetwork() {}
    NeuralNetwork(std::vector<int> siz, std::vector<int> flag) {
        assert(siz.size() == flag.size());
        ly.resize(siz.size());
        ly[0].set_input_layer(siz[0]);
        for (int i = 1; i < siz.size(); i++) {
            ly[i].set_connect_layer(ly[i - 1], siz[i], flag[i]);
            ly[i].weight_initialize();
        }
        int maxd = 0;
        for (auto &i : siz)maxd = std::max(maxd, i);
        dw.resize({(int)siz.size(), maxd, maxd});
    }
    void show()const {
        for (int i = 0; i < ly.size(); i++) {
            if (i == 0) {
                std::cout << " 0: input layer" << std::endl;
            } else {
                std::cout << " " << i << ": layer" << std::endl;
                ly[i].show();
            }
        }
    }
    Vector predict(const Vector &x) {
        assert(x.size() == ly[0].size());
        ly[0].value = ly[0].activeValue = x;
        for (int i = 1; i < ly.size(); i++) {
            ly[i].calculate(ly[i - 1].activeValue);
        }
        return ly.back().activeValue;
    }
    void calculate_diff(int I, int J, int K) {
        double temp;
        for (int j = 0; j < ly[I].size(); j++) {
            temp = active_function_diff(ly[I].activeFlag, ly[I].value[j]);
            ly[I].diffValue[j] = (j == J) ? (ly[I - 1].value[K] * temp) : 0;
        }
        for (int i = I + 1; i < ly.size(); i++) {
            for (int j = 0; j < ly[i].size(); j++) {
                temp = active_function_diff(ly[i].activeFlag, ly[i].value[j]);
                ly[i].diffValue[j] = temp * Dot(ly[i].w[j], ly[i - 1].diffValue);
            }
        }
    }
    void train(const Vector &x, const Vector &y) {
        assert(x.size() == ly[0].size());
        predict(x);
        for (int i = 1; i < ly.size(); i++) {
            for (int j = 0; j < ly[i].size(); j++) {
                for (int k = 0; k < ly[i - 1].size(); k++) {
                    calculate_diff(i, j, k);
                    dw(i, j, k) = Dot(ly.back().activeValue - y, ly.back().diffValue);
                }
            }
        }
        for (int i = 1; i < ly.size(); i++) {
            for (int j = 0; j < ly[i].size(); j++) {
                for (int k = 0; k < ly[i - 1].size(); k++) {
                    ly[i].w[j][k] -= dw(i, j, k) * eta;
                }
            }
        }
    }
};
#endif
