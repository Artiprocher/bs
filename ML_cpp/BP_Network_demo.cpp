#include <bits/stdc++.h>
#include "ML_Model.h"
#define rep(i,a,b) for(int i=(a);i<=(int)(b);i++)
using namespace std;
using namespace OLD;
typedef long long ll;

CSV_Reader csv_reader;
BP_Network net;
DataSet trainx, trainy, testx, testy;

void show_image(const Vector &a) {
    rep(i, 0, 783) {
        cout << (a[i] > 0.5 ? "*" : " ");
        if ((i + 1) % 28 == 0) cout << endl;
    }
    cout << endl;
}
void judge(const DataSet &testx, const DataSet &testy) {
    int all = testx.data.size(), ac = 0;
    rep(it, 0, all - 1) {
        Vector a = net.predict(testx.data[it]);
        int ans = 0;
        rep(i, 1, 9) {
            if (a[i] > a[ans]) ans = i;
        }
        if (testy.data[it][ans] > 0.5) ac++;
    }
    cout << (ac * 1.0 / all) << endl;
}

int main() {
    // read data
    cout << "Reading data" << endl;
    csv_reader.open("train.csv");
    //csv_reader.shuffle();
    int split_position = 30000;
    csv_reader.export_number_data(1, split_position, 1, 784, trainx);
    csv_reader.export_onehot_data(1, split_position, 0, trainy);
    csv_reader.export_number_data(split_position + 1, 42000, 1, 784, testx);
    csv_reader.export_onehot_data(split_position + 1, 42000, 0, testy);
    csv_reader.close();
    rep(i, 0, trainx.data.size() - 1) trainx.data[i] *= 1.0 / 255;
    rep(i, 0, testx.data.size() - 1) testx.data[i] *= 1.0 / 255;
    // model init
    net.init({784, 100, 10}, {CONSTANT, SIGMOID, SIGMOID});
    net.eta = 0.5;
    // train
    cout << "Training model" << endl;
    judge(testx, testy);
    ll epoch = 10000, goal = 1;
    rep(it, 1, epoch) {
        int idx = randint(0, split_position - 1);
        net.train(trainx.data[idx], trainy.data[idx]);
        if (it * 100 >= epoch * goal) {
            cout << it * 100.0 / epoch << "%" << endl;
            if (goal % 100 == 0) {
                cout << "accuracy:";
                judge(testx, testy);
            }
            goal++;
        }
    }
    return 0;
}
/*

*///
