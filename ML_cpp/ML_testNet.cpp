#include "ML_Model.h"
#define rep(i,a,b) for(int i=(int)a;i<=(int)b;i++)
typedef long long ll;

namespace net{
    const int num1=6;
    double eta=0.05;
    DenseLayer<784> I0;
    ConvLayer<28,28,5,5> C1[num1];
    MaxPoolLayer<24,24,2,2> S2[num1];
    DenseLayer<144> D3[num1];
    DenseLayer<10> L4;
    ComplateEdge<144,10> D3_L4[num1];
    auto loss=softmax_crossEntropy;
    void init(){
        //cout<<fixed<<setprecision(2);
        for(int i=0;i<num1;i++)D3[i]=DenseLayer<144>(sigmoid,sigmoid_diff);
        for(int i=0;i<num1;i++)D3[i].threshold_flag=0;
        for(int i=0;i<num1;i++)D3_L4[i]=full_connect(S2[i],L4);
    }
    Vector predict(const Vector &x){
        //清理
        I0.clear();
        for(int i=0;i<num1;i++)C1[i].clear();
        for(int i=0;i<num1;i++)S2[i].clear();
        for(int i=0;i<num1;i++)D3[i].clear();
        L4.clear();
        //正向传值
        each_index(i,x)I0.out_val[i]=x[i];
        for(int i=0;i<num1;i++)push_forward(I0,C1[i]);
        for(int i=0;i<num1;i++)C1[i].forward_solve();
        for(int i=0;i<num1;i++)push_forward(C1[i],S2[i]);
        for(int i=0;i<num1;i++)S2[i].forward_solve();
        for(int i=0;i<num1;i++)push_forward(S2[i],D3[i]);
        for(int i=0;i<num1;i++)D3[i].forward_solve();
        for(int i=0;i<num1;i++)push_forward(D3[i],L4,D3_L4[i]);
        L4.forward_solve();
        //导出结果
        static Vector y(L4.output_size,0);
        for(int i=0;i<L4.output_size;i++){
            y[i]=L4.out_val[i];
        }
        return y;
    }
    void train(const Vector &x,const Vector &y){
        //正向传值
        Vector y_=predict(x);
        //逆向传值
        Vector2Array(loss(y,y_),L4.in_diff);
        L4.backward_solve();
        for(int i=0;i<num1;i++)push_backward(D3[i],L4,D3_L4[i],eta);
        for(int i=0;i<num1;i++)D3[i].backward_solve();
        for(int i=0;i<num1;i++)push_backward(S2[i],D3[i]);
        for(int i=0;i<num1;i++)S2[i].backward_solve();
        for(int i=0;i<num1;i++)push_backward(C1[i],S2[i]);
        for(int i=0;i<num1;i++)C1[i].backward_solve();
        //更新权重
        I0.update_w(eta);
        for(int i=0;i<num1;i++)C1[i].update_w(eta);
        for(int i=0;i<num1;i++)S2[i].update_w(eta);
        for(int i=0;i<num1;i++)D3[i].update_w(eta);
        L4.update_w(eta);
    }
}

/*namespace net{
    double eta=0.05;
    DenseLayer<28*28> I0;
    ConvLayer<28,28,3,3> C1[32];
    DenseLayer<26*26> D2[32];
    MaxPoolLayer<26,26,2,2> P3[32];
    ConvLayer<13,13,4,4> C4[64];
    DenseLayer<10*10> D5[64];
    MaxPoolLayer<10,10,2,2> P6[64];
    DenseLayer<60> D7(relu,relu_diff);
    DenseLayer<10> D8(sigmoid,sigmoid_diff);
    ComplateEdge<5*5,60> P6_D7[64];
    ComplateEdge<60,10> D7_D8;
    auto loss=crossEntropy;
    void init(){
        for(int i=0;i<32;i++)D2[i]=DenseLayer<26*26>(relu,relu_diff);
        for(int i=0;i<64;i++)D5[i]=DenseLayer<10*10>(relu,relu_diff);
        for(int i=0;i<64;i++)P6_D7[i]=full_connect(P6[i],D7);
        D7_D8=full_connect(D7,D8);
    }
    Vector predict(const Vector &x){
        //清理
        I0.clear();
        rep(i,0,31)C1[i].clear(),D2[i].clear(),P3[i].clear();
        rep(i,0,63)C4[i].clear(),D5[i].clear(),P6[i].clear();
        D7.clear();
        D8.clear();
        //正向传值
        each_index(i,x)I0.out_val[i]=x[i];
        rep(i,0,31)push_forward(I0,C1[i]),C1[i].forward_solve();
        rep(i,0,31)push_forward(C1[i],D2[i]),D2[i].forward_solve();
        rep(i,0,31)push_forward(D2[i],P3[i]),P3[i].forward_solve();
        rep(i,0,31){
            push_forward(P3[i],C4[i*2]);
            C4[i*2].forward_solve();
            push_forward(P3[i],C4[i*2+1]);
            C4[i*2+1].forward_solve();
        }
        rep(i,0,63)push_forward(C4[i],D5[i]),D5[i].forward_solve();
        rep(i,0,63)push_forward(D5[i],P6[i]),P6[i].forward_solve();
        rep(i,0,63)push_forward(P6[i],D7,P6_D7[i]);
        D7.forward_solve();
        push_forward(D7,D8,D7_D8);
        D8.forward_solve();
        //导出结果
        static Vector y(D8.output_size,0);
        for(int i=0;i<D8.output_size;i++){
            y[i]=D8.out_val[i];
        }
        return y;
    }
    void train(const Vector &x,const Vector &y){
        //正向传值
        Vector y_=predict(x);
        //逆向传值
        Vector2Array(loss(y,y_),D8.in_diff);
        D8.backward_solve();
        push_backward(D7,D8,D7_D8,eta);
        D7.backward_solve();
        rep(i,0,63)push_backward(P6[i],D7,P6_D7[i],eta),P6[i].backward_solve();
        rep(i,0,63)push_backward(D5[i],P6[i]),D5[i].backward_solve();
        rep(i,0,63)push_backward(C4[i],D5[i]),C4[i].backward_solve();
        rep(i,0,31){
            push_backward(P3[i],C4[i*2]);
            push_backward(P3[i],C4[i*2+1]);
            P3[i].backward_solve();
        }
        rep(i,0,31)push_backward(D2[i],P3[i]),D2[i].backward_solve();
        rep(i,0,31)push_backward(C1[i],D2[i]),C1[i].backward_solve();
        //更新权重
        I0.update_w(eta);
        rep(i,0,31)C1[i].update_w(eta),D2[i].update_w(eta),P3[i].update_w(eta);
        rep(i,0,63)C4[i].update_w(eta),D5[i].update_w(eta),P6[i].update_w(eta);
        D7.update_w(eta);
        D8.update_w(eta);
    }
}*/

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
    //net2::predict(trainx.data[0]);
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
