#define DEBUG
#include "ML_Model.h"
#define rep(i,a,b) for(int i=(int)a;i<=(int)b;i++)
typedef long long ll;

const int C=3,H=32,W=32;

/*namespace net6{
    const int N=6;
    ParameterList PL;
    ExpandParallel< DenseLayer<32*32*C>,     N   > IN;
    Parallel<       ConvLayer<32,32,5,5>,    C*N > C1;
    Parallel<       MaxPoolLayer<28,28,2,2>, C*N > S1;
    Parallel<       DenseLayer<14*14>,       C*N > D1;
    Parallel<       ConvLayer<14,14,5,5>,    C*N > C2;
    Parallel<       MaxPoolLayer<10,10,2,2>, C*N > S2;
    DenseLayer<5*5*C*N>                            D2(sigmoid,sigmoid_diff);
    DenseLayer<100>                                L3(sigmoid,sigmoid_diff);
    DropoutLayer<100>                              Dr(0);
    DenseLayer<100>                                OU;
    auto D2_L3=full_connect(D2,L3);
    auto Dr_OU=full_connect(Dr,OU);
    auto loss=softmax_crossEntropy;
    Optimazer::Adam OP;
    void load(const char *file_name) {
        ifstream f;
        f.open(file_name,ios::in);
        PL.load(f);
        OP.load(f);
        f.close();
    }
    void save(const char *file_name) {
        ofstream f;
        f.open(file_name,ios::out);
        PL.save(f);
        OP.save(f);
        f.close();
    }
    void init(){
        for(int i=0;i<C*N;i++)D1[i]=DenseLayer<14*14>(sigmoid,sigmoid_diff);
        IN.get_parameters(PL);
        C1.get_parameters(PL);
        S1.get_parameters(PL);
        D1.get_parameters(PL);
        C2.get_parameters(PL);
        S2.get_parameters(PL);
        D2.get_parameters(PL);
        L3.get_parameters(PL);
        Dr.get_parameters(PL);
        OU.get_parameters(PL);
        D2_L3.get_parameters(PL);
        Dr_OU.get_parameters(PL);
        cout<<"number of parameters: "<<PL.w.size()<<endl;
        OP.init(PL);
    }
    Vector predict(const Vector &x){
        assert(x.size()==C*H*W);
        IN.clear();
        C1.clear();
        S1.clear();
        D1.clear();
        C2.clear();
        S2.clear();
        D2.clear();
        L3.clear();
        Dr.clear();
        OU.clear();
        Vector2Array(x,IN.in_val);
        IN.forward_solve();
        push_forward(IN,C1);
        C1.forward_solve();
        push_forward(C1,S1);
        S1.forward_solve();
        push_forward(S1,D1);
        D1.forward_solve();
        push_forward(D1,C2);
        C2.forward_solve();
        push_forward(C2,S2);
        S2.forward_solve();
        push_forward(S2,D2);
        D2.forward_solve();
        push_forward(D2,L3,D2_L3);
        L3.forward_solve();
        push_forward(L3,Dr);
        Dr.forward_solve();
        push_forward(Dr,OU,Dr_OU);
        OU.forward_solve();
        return OU.out_val.Array2Vector();
    }
    void train(const Vector &x,const Vector &y){
        assert(y.size()==100);
        Dr.set_drop_probability(0.5);
        Vector y_=predict(x);
        Vector2Array(loss(y,y_),OU.in_diff);
        OU.backward_solve();
        push_backward(Dr,OU,Dr_OU);
        Dr.backward_solve();
        push_backward(L3,Dr);
        L3.backward_solve();
        push_backward(D2,L3,D2_L3);
        D2.backward_solve();
        push_backward(S2,D2);
        S2.backward_solve();
        push_backward(C2,S2);
        C2.backward_solve();
        push_backward(D1,C2);
        D1.backward_solve();
        push_backward(S1,D1);
        S1.backward_solve();
        push_backward(C1,S1);
        C1.backward_solve();
        OP.iterate(PL);
        Dr.set_drop_probability(0);
    }
}*/

namespace net7{
    const int N=6;
    ParameterList PL;
    ExpandParallel< DenseLayer<32*32*C>,     N   > IN;
    Parallel<       ConvLayer<32,32,5,5>,    C*N > C1;
    Parallel<       MaxPoolLayer<28,28,2,2>, C*N > S1;
    Parallel<       DenseLayer<14*14>,       C*N > D1;
    Parallel<       ConvLayer<14,14,5,5>,    C*N > C2;
    Parallel<       MaxPoolLayer<10,10,2,2>, C*N > S2;
    DenseLayer<5*5*C*N>                            D2(sigmoid,sigmoid_diff);
    DenseLayer<100>                                L3(sigmoid,sigmoid_diff);
    DropoutLayer<100>                              Dr(0);
    DenseLayer<100>                                OU;
    auto D2_L3=full_connect(D2,L3);
    auto Dr_OU=full_connect(Dr,OU);
    auto loss=softmax_crossEntropy;
    Optimazer::Adam OP;
    void load(const char *file_name) {
        ifstream f;
        f.open(file_name,ios::in);
        PL.load(f);
        OP.load(f);
        f.close();
    }
    void save(const char *file_name) {
        ofstream f;
        f.open(file_name,ios::out);
        PL.save(f);
        OP.save(f);
        f.close();
    }
    void init(){
        for(int i=0;i<C*N;i++)D1[i]=DenseLayer<14*14>(sigmoid,sigmoid_diff);
        IN.get_parameters(PL);
        C1.get_parameters(PL);
        S1.get_parameters(PL);
        D1.get_parameters(PL);
        C2.get_parameters(PL);
        S2.get_parameters(PL);
        D2.get_parameters(PL);
        L3.get_parameters(PL);
        Dr.get_parameters(PL);
        OU.get_parameters(PL);
        D2_L3.get_parameters(PL);
        Dr_OU.get_parameters(PL);
        cout<<"number of parameters: "<<PL.w.size()<<endl;
        OP.init(PL);
    }
    Vector predict(const Vector &x){
        assert(x.size()==C*H*W);
        IN.clear();
        C1.clear();
        S1.clear();
        D1.clear();
        C2.clear();
        S2.clear();
        D2.clear();
        L3.clear();
        Dr.clear();
        OU.clear();
        Vector2Array(x,IN.in_val);
        IN.forward_solve();
        push_forward(IN,C1);
        C1.forward_solve();
        push_forward(C1,S1);
        S1.forward_solve();
        push_forward(S1,D1);
        D1.forward_solve();
        push_forward(D1,C2);
        C2.forward_solve();
        push_forward(C2,S2);
        S2.forward_solve();
        push_forward(S2,D2);
        D2.forward_solve();
        push_forward(D2,L3,D2_L3);
        L3.forward_solve();
        push_forward(L3,Dr);
        Dr.forward_solve();
        push_forward(Dr,OU,Dr_OU);
        OU.forward_solve();
        return OU.out_val.Array2Vector();
    }
    void train(const Vector &x,const Vector &y){
        assert(y.size()==100);
        Dr.set_drop_probability(0.5);
        Vector y_=predict(x);
        Vector2Array(loss(y,y_),OU.in_diff);
        OU.backward_solve();
        push_backward(Dr,OU,Dr_OU);
        Dr.backward_solve();
        push_backward(L3,Dr);
        L3.backward_solve();
        push_backward(D2,L3,D2_L3);
        D2.backward_solve();
        push_backward(S2,D2);
        S2.backward_solve();
        push_backward(C2,S2);
        C2.backward_solve();
        push_backward(D1,C2);
        D1.backward_solve();
        push_backward(S1,D1);
        S1.backward_solve();
        push_backward(C1,S1);
        C1.backward_solve();
        OP.iterate(PL);
        Dr.set_drop_probability(0);
    }
}

#define net net7

void judge(const DataSet &testx, const DataSet &testy) {
    int all = testx.data.size(), ac = 0;
    rep(it, 0, all - 1) {
        Vector a = net::predict(testx.data[it]);
        int ans = 0;
        rep(i, 1, (int)a.size()-1) {
            if (a[i] > a[ans]) ans = i;
        }
        if (testy.data[it][ans] > 0.5) ac++;
    }
    cout << (ac * 1.0 / all) << endl;
}

CSV_Reader csv_reader;
DataSet trainx,trainy,testx,testy;

void demo(){
    // read data
    cout << "Reading data" << endl;
    csv_reader.open("E:/coil100/data3x32x32.csv");
    //csv_reader.describe();
    csv_reader.shuffle();
    int split_position = 5000;
    csv_reader.export_number_data(0, split_position-1, 0, C*W*H-1, trainx);
    csv_reader.export_onehot_data(0, split_position-1, C*W*H, trainy);
    csv_reader.export_number_data(split_position, csv_reader.size()[0]-1, 0, C*W*H-1, testx);
    csv_reader.export_onehot_data(split_position, csv_reader.size()[0]-1, C*W*H, testy);
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
void demo_nosplit(){
    // read data
    cout << "Reading data" << endl;
    csv_reader.open("E:/coil100/data3x32x32.csv");
    //csv_reader.describe();
    csv_reader.shuffle();
    csv_reader.export_number_data(0, csv_reader.size()[0]-1, 0, C*W*H-1, trainx);
    csv_reader.export_onehot_data(0, csv_reader.size()[0]-1, C*W*H, trainy);
    csv_reader.close();
    rep(i, 0, trainx.data.size() - 1) trainx.data[i] *= 1.0 / 255;
    // train
    cout << "Training model" << endl;
    net::init();
    //net::load("net7_parameters.ini");
    judge(trainx, trainy);
    int T=500;
    while(T--){
        ll epoch = 10000;
        rep(it, 1, epoch) {
            int idx = randint(0, trainx.data.size()-1);
            net::train(trainx.data[idx], trainy.data[idx]);
        }
        judge(trainx, trainy);
        net::save("net8_parameters.ini");
    }
}

int main() {
    demo_nosplit();
    return 0;
}
/*

*///
