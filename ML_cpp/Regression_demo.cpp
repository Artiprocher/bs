#include "ML_Model.h"

#define LinearRegression(L,N,M)                          \
namespace L{                                             \
    double eta=0.001;                                    \
    ActiveLayer<N> input_layer;                          \
    ActiveLayer<M> output_layer(constant,constant_diff); \
    auto E=full_connect(input_layer,output_layer);       \
    auto loss=mse;                                       \
    void init(){                                         \
        ;                                                \
    }                                                    \
    Vector predict(const Vector &x){                     \
        input_layer.clear();                             \
        output_layer.clear();                            \
        each_index(i,x)input_layer.out_val[i]=x[i];      \
        push_forward(input_layer,output_layer,E);        \
        output_layer.forward_solve();                    \
        static Vector y(output_layer.output_size,0);     \
        for(int i=0;i<output_layer.output_size;i++){     \
            y[i]=output_layer.out_val[i];                \
        }                                                \
        return y;                                        \
    }                                                    \
    void train(const Vector &x,const Vector &y){         \
        Vector y_=predict(x);                            \
        Vector2Array(loss(y,y_),output_layer.diff_val);  \
        push_backward(input_layer,output_layer,E,eta);   \
        output_layer.update_w(eta);                      \
    }                                                    \
}

#define LogitRegression(L,N,M)                           \
namespace L{                                             \
    double eta=0.1;                                      \
    ActiveLayer<N> input_layer;                          \
    ActiveLayer<M> output_layer(sigmoid,sigmoid_diff);   \
    auto E=full_connect(input_layer,output_layer);       \
    auto loss=mse;                                       \
    void init(){                                         \
        ;                                                \
    }                                                    \
    Vector predict(const Vector &x){                     \
        input_layer.clear();                             \
        output_layer.clear();                            \
        each_index(i,x)input_layer.out_val[i]=x[i];      \
        push_forward(input_layer,output_layer,E);        \
        output_layer.forward_solve();                    \
        static Vector y(output_layer.output_size,0);     \
        for(int i=0;i<output_layer.output_size;i++){     \
            y[i]=output_layer.out_val[i];                \
        }                                                \
        return y;                                        \
    }                                                    \
    void train(const Vector &x,const Vector &y){         \
        Vector y_=predict(x);                            \
        Vector2Array(loss(y,y_),output_layer.diff_val);  \
        push_backward(input_layer,output_layer,E,eta);   \
        output_layer.update_w(eta);                      \
    }                                                    \
}

LogitRegression(L,784,10);

CSV_Reader csv_reader;
DataSet trainx,trainy,testx,testy,x,y,t;

void show_image(const Vector &a){
    rep(i,0,783){
        cout<<(a[i]>0.5?"*":" ");
        if((i+1)%28==0)cout<<endl;
    }
    cout<<endl;
}
void judge1(const DataSet &testx, const DataSet &testy) {
    int all = testx.data.size(), ac = 0;
    rep(it, 0, all - 1) {
        Vector a = L::predict(testx.data[it]);
        int ans = 0;
        rep(i, 1, 9) {
            if (a[i] > a[ans]) ans = i;
        }
        if (testy.data[it][ans] > 0.5) ac++;
    }
    cout << (ac * 1.0 / all) << endl;
}
int main(){
    // read data
    cout << "Reading data" << endl;
    csv_reader.open("train.csv");
    csv_reader.shuffle();
    int split_position = 30000;
    csv_reader.export_number_data(0, split_position-1, 1, 784, trainx);
    csv_reader.export_onehot_data(0, split_position-1, 0, trainy);
    csv_reader.export_number_data(split_position, 42000-1, 1, 784, testx);
    csv_reader.export_onehot_data(split_position, 42000-1, 0, testy);
    csv_reader.close();
    rep(i, 0, trainx.data.size() - 1) trainx.data[i] *= 1.0 / 255;
    rep(i, 0, testx.data.size() - 1) testx.data[i] *= 1.0 / 255;
    // train
    cout << "Training model" << endl;
    L::init();
    judge1(testx, testy);
    ll epoch = 1000000, goal = 1;
    rep(it, 1, epoch) {
        int idx = randint(0, split_position - 1);
        L::train(trainx.data[idx], trainy.data[idx]);
        if (it * 100 >= epoch * goal) {
            cout << it * 100.0 / epoch << "%" << endl;
            if (goal % 10 == 0) {
                cout << "accuracy:";
                judge1(testx, testy);
            }
            goal++;
        }
    }
    return 0;
}