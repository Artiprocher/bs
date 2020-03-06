#include "ML_Model.h"
#define rep(i,a,b) for(int i=(int)a;i<=(int)b;i++)
typedef long long ll;

namespace net{
    double eta=0.1;
    ParameterList PL;
    DenseLayer<784> input_layer;
    DenseLayer<30> hidden_layer(sigmoid,sigmoid_diff);
    DenseLayer<10> output_layer(sigmoid,sigmoid_diff);
    ComplateEdge<784,30> E1;
    ComplateEdge<30,10> E2;
    auto loss=mse;
    Optimazer::GradientDescent GD(eta);
    void init(){
        input_layer.get_parameters(PL);
        hidden_layer.get_parameters(PL);
        output_layer.get_parameters(PL);
        E1.get_parameters(PL);
        E2.get_parameters(PL);
    }
    Vector predict(const Vector &x){
        /*清理*/
        input_layer.clear();
        hidden_layer.clear();
        output_layer.clear();
        /*正向传值*/
        each_index(i,x)input_layer.out_val[i]=x[i];
        push_forward(input_layer,hidden_layer,E1);
        hidden_layer.forward_solve();
        push_forward(hidden_layer,output_layer,E2);
        output_layer.forward_solve();
        /*导出结果*/
        static Vector y(output_layer.output_size,0);
        for(int i=0;i<output_layer.output_size;i++){
            y[i]=output_layer.out_val[i];
        }
        return y;
    }
    void train(const Vector &x,const Vector &y){
        /*正向传值*/
        Vector y_=predict(x);
        /*逆向传值*/
        Vector2Array(loss(y,y_),output_layer.in_diff);
        output_layer.backward_solve();
        push_backward(hidden_layer,output_layer,E2);
        hidden_layer.backward_solve();
        push_backward(input_layer,hidden_layer,E1);
        /*更新权重*/
        GD.iterate(PL);
    }
}

CSV_Reader csv_reader;
DataSet trainx,trainy,testx,testy;

void show_image(const Vector &a){
    rep(i,0,783){
        cout<<(a[i]>0.5?"*":" ");
        if((i+1)%28==0)cout<<endl;
    }
    cout<<endl;
}
void judge(const DataSet &testx, const DataSet &testy) {
    int all = testx.data.size(), ac = 0;
    rep(it, 0, all - 1) {
        Vector a = net::predict(testx.data[it]);
        int ans = 0;
        rep(i, 1, 9) {
            if (a[i] > a[ans]) ans = i;
        }
        if (testy.data[it][ans] > 0.5) ac++;
    }
    cout << (ac * 1.0 / all) << endl;
}
void demo(){
    // read data
    cout << "Reading data" << endl;
    csv_reader.open("digit/train.csv");
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
    net::init();
    judge(testx, testy);
    while(1){
        ll epoch = 10000;
        rep(it, 1, epoch) {
            int idx = randint(0, split_position - 1);
            net::train(trainx.data[idx], trainy.data[idx]);
        }
        judge(testx, testy);
    }
}

int main() {
    demo();
    return 0;
}
/*

*///
