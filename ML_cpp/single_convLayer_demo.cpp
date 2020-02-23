#include "ML_Model.h"
#define rep(i,a,b) for(int i=(int)a;i<=(int)b;i++)
typedef long long ll;

namespace net2{
    ParameterList PL;
    const int num1=10;
    double eta=0.05;
    DenseLayer<784> I0;
    ConvLayer<28,28,5,5> C1[num1];
    MaxPoolLayer<24,24,2,2> S2[num1];
    DenseLayer<144> D3[num1];
    DenseLayer<10> L4;
    ComplateEdge<144,10> D3_L4[num1];
    auto loss=softmax_crossEntropy;
    Optimazer::GradientDescent GD(eta);
    void init(){
        for(int i=0;i<num1;i++)D3[i]=DenseLayer<144>(sigmoid,sigmoid_diff);
        I0.get_parameters(PL);
        for(int i=0;i<num1;i++)C1[i].get_parameters(PL);
        for(int i=0;i<num1;i++)S2[i].get_parameters(PL);
        for(int i=0;i<num1;i++)D3[i].get_parameters(PL);
        for(int i=0;i<num1;i++)D3_L4[i].get_parameters(PL);
        L4.get_parameters(PL);
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
        for(int i=0;i<num1;i++)push_backward(D3[i],L4,D3_L4[i]);
        for(int i=0;i<num1;i++)D3[i].backward_solve();
        for(int i=0;i<num1;i++)push_backward(S2[i],D3[i]);
        for(int i=0;i<num1;i++)S2[i].backward_solve();
        for(int i=0;i<num1;i++)push_backward(C1[i],S2[i]);
        for(int i=0;i<num1;i++)C1[i].backward_solve();
        //更新权重
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
    csv_reader.open("digit/train.csv");
    csv_reader.shuffle();
    int all = 42000;
    csv_reader.export_number_data(0, all-1, 1, 784, trainx);
    csv_reader.export_onehot_data(0, all-1, 0, trainy);
    csv_reader.close();
    rep(i, 0, trainx.data.size() - 1) trainx.data[i] *= 1.0 / 255;
    csv_reader.open("digit/test.csv");
    csv_reader.export_number_data(0, 28000-1, 0, 783, testx);
    rep(i, 0, testx.data.size() - 1) testx.data[i] *= 1.0 / 255;
    csv_reader.close();
    // train
    cout << "Training model" << endl;
    net2::init();
    rep(it,1,50){
        int epoch = 10000;
        rep(it, 1, epoch) {
            int idx = randint(0, all - 1);
            net2::train(trainx.data[idx], trainy.data[idx]);
        }
        judge2(trainx, trainy);
    }
    std::ofstream fout;
    fout.open("digit/result.txt", std::ios::out);
    for(int i=0;i<28000;i++){
        Vector y=net2::predict(testx.data[i]);
        int ans=0;
        for(int i=1;i<10;i++)if(y[i]>y[ans])ans=i;
        fout<<ans<<endl;
    }
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
    net2::init();
    //net2::predict(trainx.data[0]);
    judge2(testx, testy);
    while(1){
        ll epoch = 1000;
        rep(it, 1, epoch) {
            int idx = randint(0, split_position - 1);
            net2::train(trainx.data[idx], trainy.data[idx]);
        }
        judge2(testx, testy);
    }
}

int main() {
    demo();
    return 0;
}
/*

*///
