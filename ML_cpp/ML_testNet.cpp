#define DEBUG
#include "ML_Model.h"
#define rep(i,a,b) for(int i=(int)a;i<=(int)b;i++)
typedef long long ll;

//单层卷积池化
/*namespace net2{
    ParameterList PL;
    const int N=10;
    DenseLayer<784> I0;
    ExpandParallel< ConvLayer<28,28,5,5>,N > C1;
    Parallel< MaxPoolLayer<24,24,2,2>,N > S2;
    Parallel< DenseLayer<144>,N > D3;
    ComplateEdge<144*N,10> D3_L4;
    DenseLayer<10> L4;
    auto loss=softmax_crossEntropy;
    //Optimazer::GradientDescent OP(0.05);
    Optimazer::Adam OP;
    void init(){
        for(int i=0;i<N;i++)D3[i]=DenseLayer<144>(sigmoid,sigmoid_diff);
        I0.get_parameters(PL);
        C1.get_parameters(PL);
        S2.get_parameters(PL);
        D3.get_parameters(PL);
        D3_L4.get_parameters(PL);
        L4.get_parameters(PL);
        OP.init(PL);
    }
    Vector predict(const Vector &x){
        //清理
        I0.clear();
        C1.clear();
        S2.clear();
        D3.clear();
        L4.clear();
        //正向传值
        each_index(i,x)I0.out_val[i]=x[i];
        push_forward(I0,C1);
        C1.forward_solve();
        push_forward(C1,S2);
        S2.forward_solve();
        push_forward(S2,D3);
        D3.forward_solve();
        push_forward(D3,L4,D3_L4);
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
        push_backward(D3,L4,D3_L4);
        D3.backward_solve();
        push_backward(S2,D3);
        S2.backward_solve();
        push_backward(C1,S2);
        C1.backward_solve();
        //更新权重
        OP.iterate(PL);
    }
}*/

//dropout测试
/*namespace net3{
    ParameterList PL;
    DenseLayer<784> L0;
    ComplateEdge<784,30> L0_L1;
    DenseLayer<30> L1(sigmoid,sigmoid_diff);
    DropoutLayer<30> L2(0);
    ComplateEdge<30,10> L2_L3;
    DenseLayer<10> L3;
    auto loss=softmax_crossEntropy;
    //Optimazer::GradientDescent OP(0.05);
    Optimazer::Adam OP;
    void init(){
        L0.get_parameters(PL);L1.get_parameters(PL);L2.get_parameters(PL);L3.get_parameters(PL);
        L0_L1.get_parameters(PL);L2_L3.get_parameters(PL);
        OP.init(PL);
    }
    Vector predict(const Vector &x){
        L0.clear();L1.clear();L2.clear();L3.clear();
        each_index(i,x)L0.out_val[i]=x[i];
        push_forward(L0,L1,L0_L1);L1.forward_solve();
        push_forward(L1,L2);L2.forward_solve();
        push_forward(L2,L3,L2_L3);L3.forward_solve();
        return L3.out_val.Array2Vector();
    }
    void train(const Vector &x,const Vector &y){
        Vector y_=predict(x);
        Vector2Array(loss(y,y_),L3.in_diff);
        L3.backward_solve();push_backward(L2,L3,L2_L3);
        L2.backward_solve();push_backward(L1,L2);
        L1.backward_solve();push_backward(L0,L1,L0_L1);
        OP.iterate(PL);
    }
}*/

//双层卷积池化
namespace net4{
    const int N=20,M=20;
    ParameterList PL;
    DenseLayer<784> IN;
    ExpandParallel< ConvLayer<28,28,5,5>,N > C1;
    Parallel< MaxPoolLayer<24,24,2,2>,N > S1;
    Parallel< DenseLayer<12*12>,N > D1;
    Parallel< ConvLayer<12,12,5,5>,M > C2;
    Parallel< MaxPoolLayer<8,8,2,2>,M > S2;
    DenseLayer<4*4*M> D2;
    DenseLayer<100> L3;
    DropoutLayer<100> Dr(0);
    DenseLayer<10> OU;
    auto D2_L3=full_connect(D2,L3);
    auto Dr_OU=full_connect(Dr,OU);
    function<Vector(Vector,Vector)> loss=softmax_crossEntropy;
    Optimazer::Adam OP;
    void init(){
        for(int i=0;i<N;i++)D1[i]=DenseLayer<12*12>(relu,relu_diff);
        D2=DenseLayer<4*4*M>(relu,relu_diff);
        L3=DenseLayer<100>(sigmoid,sigmoid_diff);
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
        //push_forward(D1,C2);
        for(int i=0;i<M;i++){
            for(int j=0;j<1;j++)push_forward(D1[(i+j)%N],C2[i]);
            C2[i].forward_solve();
        }
        C2.pass_out_val();
        //C2.forward_solve();
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
        //Dr.set_drop_probability(0.5);
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
        //push_backward(D1,C2);
        for(int i=0;i<M;i++){
            for(int j=0;j<1;j++)push_backward(D1[(i+j)%N],C2[i]);
        }
        for(int i=0;i<N;i++){
            D1[i].backward_solve();
        }
        D1.pass_out_diff();
        //D1.backward_solve();
        push_backward(S1,D1);
        S1.backward_solve();
        push_backward(C1,S1);
        C1.backward_solve();
        push_backward(IN,C1);
        IN.backward_solve();
        OP.iterate(PL);
        //Dr.set_drop_probability(0);
    }
}

//LeNet-5*
/*namespace net5{
    ParameterList PL;
    DenseLayer<784> IN;
    ExpandParallel< ConvLayer<28,28,5,5>,6 > C1;
    Parallel< MaxPoolLayer<24,24,2,2>,6 > S1;
    Parallel< DenseLayer<12*12>,6 > D1;
    vector<int> ConnectTable[16]={
        {0,1,2},
        {1,2,3},
        {2,3,4},
        {3,4,5},
        {4,5,0},
        {5,0,1},
        {0,1,2,3},
        {1,2,3,4},
        {2,3,4,5},
        {3,4,5,0},
        {4,5,0,1},
        {5,0,1,2},
        {0,1,3,4},
        {1,2,4,5},
        {0,2,3,5},
        {0,1,2,3,4,5}
    };
    Parallel< ConvLayer<12,12,5,5>,16 > C2;
    Parallel< MaxPoolLayer<8,8,2,2>,16 > S2;
    Parallel< DenseLayer<4*4>,16 > D2;
    Parallel< ConvLayer<4,4,4,4>,16*100 > C3;
    Parallel< DenseLayer<1*1>,100 > D3;
    DenseLayer<10> OU;
    auto D3_OU=full_connect(D3,OU);
    function<Vector(Vector,Vector)> loss=softmax_crossEntropy;
    Optimazer::Adam OP;
    void init(){
        for(int i=0;i<6;i++)D1[i]=DenseLayer<12*12>(sigmoid,sigmoid_diff);
        for(int i=0;i<16;i++)D2[i]=DenseLayer<4*4>(sigmoid,sigmoid_diff);
        for(int i=0;i<100;i++)D3[i]=DenseLayer<1*1>(sigmoid,sigmoid_diff);
        IN.get_parameters(PL);
        C1.get_parameters(PL);
        S1.get_parameters(PL);
        D1.get_parameters(PL);
        C2.get_parameters(PL);
        S2.get_parameters(PL);
        D2.get_parameters(PL);
        C3.get_parameters(PL);
        D3.get_parameters(PL);
        OU.get_parameters(PL);
        D3_OU.get_parameters(PL);
        OP.init(PL);
        cout<<"number of parameters: "<<PL.w.size()<<endl;
    }
    Vector predict(const Vector &x){
        IN.clear();
        C1.clear();
        S1.clear();
        D1.clear();
        C2.clear();
        S2.clear();
        D2.clear();
        C3.clear();
        D3.clear();
        OU.clear();
        Vector2Array(x,IN.in_val);
        IN.forward_solve();
        push_forward(IN,C1);
        C1.forward_solve();
        push_forward(C1,S1);
        S1.forward_solve();
        push_forward(S1,D1);
        D1.forward_solve();
        for(int i=0;i<16;i++){
            for(auto j:ConnectTable[i])push_forward(D1[j],C2[i]);
            C2[i].forward_solve();
        }
        C2.pass_out_val();
        push_forward(C2,S2);
        S2.forward_solve();
        push_forward(S2,D2);
        D2.forward_solve();
        for(int i=0;i<16;i++){
            for(int j=0;j<100;j++){
                push_forward(D2[i],C3[i*100+j]);
                C3[i*100+j].forward_solve();
            }
        }
        for(int i=0;i<16;i++){
            for(int j=0;j<100;j++){
                push_forward(C3[i*100+j],D3[j]);
            }
        }
        for(int j=0;j<100;j++)D3[j].forward_solve();
        D3.pass_out_val();
        push_forward(D3,OU,D3_OU);
        OU.forward_solve();
        return OU.out_val.Array2Vector();
    }
    void train(const Vector &x,const Vector &y){
        Vector y_=predict(x);
        Vector2Array(loss(y,y_),OU.in_diff);
        OU.backward_solve();
        push_backward(D3,OU,D3_OU);
        D3.backward_solve();
        for(int i=0;i<16;i++){
            for(int j=0;j<100;j++){
                push_backward(C3[i*100+j],D3[j]);
                C3[i*100+j].backward_solve();
            }
        }
        for(int i=0;i<16;i++){
            for(int j=0;j<100;j++){
                push_backward(D2[i],C3[i*100+j]);
            }
            D2[i].backward_solve();
        }
        D2.pass_out_diff();
        push_backward(S2,D2);
        S2.backward_solve();
        push_backward(C2,S2);
        C2.backward_solve();
        for(int i=0;i<16;i++){
            for(auto j:ConnectTable[i])push_backward(D1[j],C2[i]);
        }
        for(int i=0;i<6;i++)D1[i].backward_solve();
        D1.pass_out_diff();
        push_backward(S1,D1);
        S1.backward_solve();
        push_backward(C1,S1);
        C1.backward_solve();
        push_backward(IN,C1);
        IN.backward_solve();
        OP.iterate(PL);
    }
}*/

#define net net4

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
    //net::predict(trainx.data[0]);
    judge2(testx, testy);
    while(1){
        ll epoch = 10000;
        rep(it, 1, epoch) {
            int idx = randint(0, split_position - 1);
            net::train(trainx.data[idx], trainy.data[idx]);
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
