#define DEBUG
#include "ML_Model.h"
#define rep(i,a,b) for(int i=(int)a;i<=(int)b;i++)
typedef long long ll;

class Conv2{
public:
    static const int N=20;
    ParameterList PL;
    DenseLayer<784> IN;
    ExpandParallel< ConvLayer<28,28,5,5>,N > C1;
    Parallel< MaxPoolLayer<24,24,2,2>,N > S1;
    Parallel< DenseLayer<12*12>,N > D1;
    Parallel< ConvLayer<12,12,5,5>,N > C2;
    Parallel< MaxPoolLayer<8,8,2,2>,N > S2;
    DenseLayer<4*4*N> D2;
    DenseLayer<100> L3;
    DropoutLayer<100> Dr=DropoutLayer<100>(0);
    DenseLayer<10> OU;
    ComplateEdge<4*4*N,100> D2_L3=full_connect(D2,L3);
    ComplateEdge<100,10> Dr_OU=full_connect(Dr,OU);
    function<Vector(Vector,Vector)> loss=softmax_crossEntropy;
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
        for(int i=0;i<N;i++)D1[i]=DenseLayer<12*12>(relu,relu_diff);
        D2=DenseLayer<4*4*N>(relu,relu_diff);
        L3=DenseLayer<100>(sigmoid,sigmoid_diff);
        OU.use_bias=1;
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
    void back_propagate(){
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
        push_backward(IN,C1);
        IN.backward_solve();
    }
    void train(const Vector &x,const Vector &y){
        Dr.set_drop_probability(0.5);
        Vector y_=predict(x);
        Vector2Array(loss(y,y_),OU.in_diff);
        back_propagate();
        OP.iterate(PL);
        Dr.set_drop_probability(0);
    }
    void saliency_maps_back(const Vector &x,const Vector &y){
        Vector y_=predict(x);
        Vector2Array(y,OU.in_diff);
        back_propagate();
    }
}net;

class Conv2_{
public:
    static const int N=20;
    ParameterList PL;
    DenseLayer<784> IN;
    ExpandParallel< ConvLayer<28,28,5,5>,N > C1;
    Parallel< MaxPoolLayer<24,24,2,2>,N > S1;
    Parallel< DenseLayer<12*12>,N > D1;
    Parallel< ConvLayer<12,12,5,5>,N > C2;
    Parallel< AvePoolLayer<8,8,8,8>,N > S2;
    DenseLayer<10> OU;
    ComplateEdge<N,10> S2_OU=full_connect(S2,OU);
    function<Vector(Vector,Vector)> loss=softmax_crossEntropy;
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
    void transform(const char *file_name){
        static ParameterList ls;
        ls.clear();
        IN.get_parameters(ls);
        C1.get_parameters(ls);
        S1.get_parameters(ls);
        D1.get_parameters(ls);
        C2.get_parameters(ls);
        S2.get_parameters(ls);
        ifstream f;
        f.open(file_name,ios::in);
        ls.load(f);
        f.close();
    }
    void init(){
        for(int i=0;i<N;i++)D1[i]=DenseLayer<12*12>(relu,relu_diff);
        OU.get_parameters(PL);
        S2_OU.get_parameters(PL);
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
        push_forward(S2,OU,S2_OU);
        OU.forward_solve();
        return OU.out_val.Array2Vector();
    }
    void train(const Vector &x,const Vector &y){
        Vector y_=predict(x);
        Vector2Array(loss(y,y_),OU.in_diff);
        OU.backward_solve();
        push_backward(S2,OU,S2_OU);
        OP.iterate(PL);
    }
}net2;

CSV_Reader csv_reader;
DataSet trainx,trainy,testx,testy;

template <class type>
void save_image(type data,int h,int w,string name){
    ofstream f;
    f.open("vis/data.txt",ios::out);
    f<<fixed<<setprecision(6);
    rep(i,0,h-1){
        rep(j,0,w-1)f<<data[i*w+j]<<" \n"[j==w-1];
    }
    f.close();
    system(("python vis/digit_vis.py "+name).data());
}
template <class type>
void judge(type &model,const DataSet &testx, const DataSet &testy) {
    int all = testx.data.size(), ac = 0;
    rep(it, 0, all - 1) {
        Vector a = model.predict(testx.data[it]);
        int ans = 0;
        rep(i, 1, 9) {
            if (a[i] > a[ans]) ans = i;
        }
        if (testy.data[it][ans] > 0.5) ac++;
    }
    cout << (ac * 1.0 / all) << endl;
}
void read_data(){
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
}
void train_network(){
    cout << "Training model" << endl;
    net.init();
    int split_position = 30000;
    judge(net, testx, testy);
    int T=1000000;
    while(T--){
        int idx = randint(0, split_position - 1);
        net.train(trainx.data[idx], trainy.data[idx]);
        if(T%10000==0)judge(net, testx,testy);
    }
    net.save("net_parameters.ini");
}
void train_network2(){
    cout << "Training model2" << endl;
    net2.init();
    net2.transform("net_parameters.ini");
    int split_position = 30000;
    judge(net2, testx, testy);
    int T=1000000;
    while(T--){
        int idx = randint(0, split_position - 1);
        net2.train(trainx.data[idx], trainy.data[idx]);
        if(T%10000==0)judge(net2,testx,testy);
    }
    net2.save("net2_parameters.ini");
}
void vis_network(Vector x,Vector y){
    net.init();
    net.load("net_parameters.ini");
    //image
    save_image(x,28,28,"image");
    //saliency map
    net.saliency_maps_back(x,y);
    save_image(net.IN.out_diff,28,28,"saliency_maps");
    //occlusion sensitivity
    function<Vector(Vector)> softmax=[](Vector x){
        double sum=0;
        for(auto &i:x)i=exp(i),sum+=i;
        for(auto &i:x)i=i/sum;
        return x;
    };
    int label=0;
    rep(i,0,9)if(y[i]>0.5)label=i;
    int d=5;
    Vector m((29-d)*(29-d),0);
    rep(i,0,28-d)rep(j,0,28-d){
        Vector xx=x;
        rep(k1,i,i+d-1)rep(k2,j,j+d-1)xx[k1*28+k2]=0.5;
        m[i*(29-d)+j]=softmax(net.predict(xx))[label];
    }
    save_image(m,29-d,29-d,"occlusion_sensitivity");
    //class activation map
    net2.init();
    net2.load("net2_parameters.ini");
    string name="class_activation_map_0";
    rep(number,0,9){
        net2.predict(x);
        auto ans=net2.C2[0].out_val;
        ans.clear();
        rep(i,0,net2.N)net2.C2[i].out_val*=net2.S2_OU.w(i,number);
        rep(i,0,net2.N)ans+=net2.C2[i].out_val;
        save_image(ans,8,8,name.data());
        name[name.size()-1]++;
    }
}

int main() {
    read_data();
    //train_network();
    train_network2();
    //int idx=11;
    //vis_network(trainx.data[idx],trainy.data[idx]);
    return 0;
}
/*

*///
