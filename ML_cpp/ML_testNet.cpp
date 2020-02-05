#include "ML_Model.h"

namespace net2{
    const int num1=6,num2=6;
    double eta=0.01;
    ActiveLayer<784> I0;
    ConvLayer<28,28,5,5> C1[num1];
    MaxPoolLayer<24,24,2,2> S2[num1];
    ConvLayer<12,12,5,5> C3[num2];
    MaxPoolLayer<8,8,2,2> S4[num2];
    ActiveLayer<10> L5(sigmoid,sigmoid_diff);
    ComplateEdge<16,10> S4_L5[num2];
    auto loss=mse;
    void init(){
        for(int i=0;i<num1;i++)S2[i]=MaxPoolLayer<24,24,2,2>(sigmoid,sigmoid_diff);
        for(int i=0;i<num2;i++)S4[i]=MaxPoolLayer<8,8,2,2>(sigmoid,sigmoid_diff);
        for(int i=0;i<num2;i++)S4_L5[i]=full_connect(S4[i],L5);
    }
    Vector predict(const Vector &x){
        /*清理*/
        I0.clear();
        for(int i=0;i<num1;i++)C1[i].clear();
        for(int i=0;i<num1;i++)S2[i].clear();
        for(int i=0;i<num2;i++)C3[i].clear();
        for(int i=0;i<num2;i++)S4[i].clear();
        L5.clear();
        /*正向传值*/
        each_index(i,x)I0.out_val[i]=x[i];
        for(int i=0;i<num1;i++)push_forward(I0,C1[i]);
        for(int i=0;i<num1;i++)C1[i].forward_solve();
        for(int i=0;i<num1;i++)push_forward(C1[i],S2[i]);
        for(int i=0;i<num1;i++)S2[i].forward_solve();
        for(int i=0;i<num2;i++){
            for(int j=0;j<3;j++)push_forward(S2[(i+j)%num1],C3[i]);
        }
        for(int i=0;i<num2;i++)C3[i].forward_solve();
        for(int i=0;i<num2;i++)push_forward(C3[i],S4[i]);
        for(int i=0;i<num2;i++)S4[i].forward_solve();
        for(int i=0;i<num2;i++)push_forward(S4[i],L5,S4_L5[i]);
        L5.forward_solve();
        /*导出结果*/
        static Vector y(L5.output_size,0);
        for(int i=0;i<L5.output_size;i++){
            y[i]=L5.out_val[i];
        }
        return y;
    }
    void train(const Vector &x,const Vector &y){
        /*正向传值*/
        Vector y_=predict(x);
        /*逆向传值*/
        Vector2Array(loss(y,y_),L5.diff_val);
        for(int i=0;i<num2;i++)push_backward(S4[i],L5,S4_L5[i],eta);
        for(int i=0;i<num2;i++)push_backward(C3[i],S4[i],eta);
        for(int i=0;i<num2;i++){
            for(int j=0;j<3;j++)push_backward(S2[(i+j)%num1],C3[i],eta);
        }
        for(int i=0;i<num1;i++)push_backward(C1[i],S2[i],eta);
        for(int i=0;i<num1;i++)push_backward(I0,C1[i],eta);
        /*更新权重*/
        I0.update_w(eta);
        for(int i=0;i<num1;i++)C1[i].update_w(eta);
        for(int i=0;i<num1;i++)S2[i].update_w(eta);
        for(int i=0;i<num2;i++)C3[i].update_w(eta);
        for(int i=0;i<num2;i++)S4[i].update_w(eta);
        L5.update_w(eta);
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
void judge2(const DataSet &testx, const DataSet &testy) {
    int all = testx.data.size(), ac = 0;
    rep(it, 0, all - 1) {
        Vector a = net2::predict(testx.data[it]);
        int ans = 0;
        rep(i, 1, 9) {
            if (a[i] > a[ans]) ans = i;
        }
        if (testy.data[it][ans] > 0.5) ac++;
    }
    cout << (ac * 1.0 / all) << endl;
}
void demo2(){
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
    net2::init();
    //net2::predict(trainx.data[0]);
    judge2(testx, testy);
    while(1){
        ll epoch = 10000;
        rep(it, 1, epoch) {
            int idx = randint(0, split_position - 1);
            net2::train(trainx.data[idx], trainy.data[idx]);
        }
        judge2(testx, testy);
    }
}

int main() {
    demo2();
    return 0;
}
/*

*///
