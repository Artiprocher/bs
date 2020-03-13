#define DEBUG
#include "ML_Model.h"
#define rep(i,a,b) for(int i=(int)a;i<=(int)b;i++)
typedef long long ll;

namespace net{
    ParameterList PL;
    const int num1=10;
    DenseLayer<784> I0;
    ConvLayer<28,28,5,5> C1[num1];
    MaxPoolLayer<24,24,2,2> S2[num1];
    DenseLayer<144> D3[num1];
    DenseLayer<10> L4;
    ComplateEdge<144,10> D3_L4[num1];
    auto loss=softmax_crossEntropy;
    //Optimazer::GradientDescent OP(0.05);
    Optimazer::Adam OP;
    void load(const char *file_name) {
        ifstream f;
        f.open(file_name,ios::in);
        PL.load(f);
        f.close();
    }
    void save(const char *file_name) {
        ofstream f;
        f.open(file_name,ios::out);
        PL.save(f);
        f.close();
    }
    void init(){
        for(int i=0;i<num1;i++)D3[i]=DenseLayer<144>(sigmoid,sigmoid_diff);
        I0.get_parameters(PL);
        for(int i=0;i<num1;i++)C1[i].get_parameters(PL);
        for(int i=0;i<num1;i++)S2[i].get_parameters(PL);
        for(int i=0;i<num1;i++)D3[i].get_parameters(PL);
        for(int i=0;i<num1;i++)D3_L4[i].get_parameters(PL);
        L4.get_parameters(PL);
        OP.init(PL);
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
        for(int i=0;i<num1;i++)push_backward(I0,C1[i]);
        I0.backward_solve();
        //更新权重
        OP.iterate(PL);
    }
    void saliency_maps_back(const Vector &x,const Vector &y){
        //正向传值
        Vector y_=predict(x);
        //逆向传值
        Vector2Array(y,L4.in_diff);
        L4.backward_solve();
        for(int i=0;i<num1;i++)push_backward(D3[i],L4,D3_L4[i]);
        for(int i=0;i<num1;i++)D3[i].backward_solve();
        for(int i=0;i<num1;i++)push_backward(S2[i],D3[i]);
        for(int i=0;i<num1;i++)S2[i].backward_solve();
        for(int i=0;i<num1;i++)push_backward(C1[i],S2[i]);
        for(int i=0;i<num1;i++)C1[i].backward_solve();
        for(int i=0;i<num1;i++)push_backward(I0,C1[i]);
        I0.backward_solve();
    }
}
namespace net2{
    ParameterList PL;
    const int num1=10;
    DenseLayer<784> I0;
    ConvLayer<28,28,5,5> C1[num1];
    MaxPoolLayer<24,24,2,2> S2[num1];
    DenseLayer<144> D3[num1];
    AvePoolLayer<12,12,12,12> P4[num1];
    DenseLayer<10> L5;
    ComplateEdge<1,10> P4_L5[num1];
    auto loss=softmax_crossEntropy;
    Optimazer::Adam OP;
    void load(const char *file_name) {
        ifstream f;
        f.open(file_name,ios::in);
        PL.load(f);
        f.close();
    }
    void save(const char *file_name) {
        ofstream f;
        f.open(file_name,ios::out);
        PL.save(f);
        f.close();
    }
    void init(){
        for(int i=0;i<num1;i++)C1[i]=net::C1[i];
        for(int i=0;i<num1;i++)S2[i]=net::S2[i];
        for(int i=0;i<num1;i++)D3[i]=net::D3[i];
        for(int i=0;i<num1;i++)P4_L5[i].get_parameters(PL);
        L5.get_parameters(PL);
        OP.init(PL);
    }
    Vector predict(const Vector &x){
        //清理
        I0.clear();
        for(int i=0;i<num1;i++)C1[i].clear();
        for(int i=0;i<num1;i++)S2[i].clear();
        for(int i=0;i<num1;i++)D3[i].clear();
        for(int i=0;i<num1;i++)P4[i].clear();
        L5.clear();
        //正向传值
        each_index(i,x)I0.out_val[i]=x[i];
        for(int i=0;i<num1;i++)push_forward(I0,C1[i]);
        for(int i=0;i<num1;i++)C1[i].forward_solve();
        for(int i=0;i<num1;i++)push_forward(C1[i],S2[i]);
        for(int i=0;i<num1;i++)S2[i].forward_solve();
        for(int i=0;i<num1;i++)push_forward(S2[i],D3[i]);
        for(int i=0;i<num1;i++)D3[i].forward_solve();
        for(int i=0;i<num1;i++)push_forward(D3[i],P4[i]);
        for(int i=0;i<num1;i++)P4[i].forward_solve();
        for(int i=0;i<num1;i++)push_forward(P4[i],L5,P4_L5[i]);
        L5.forward_solve();
        //导出结果
        static Vector y(L5.output_size,0);
        for(int i=0;i<L5.output_size;i++){
            y[i]=L5.out_val[i];
        }
        return y;
    }
    void train(const Vector &x,const Vector &y){
        //正向传值
        Vector y_=predict(x);
        //逆向传值
        Vector2Array(loss(y,y_),L5.in_diff);
        L5.backward_solve();
        for(int i=0;i<num1;i++)push_backward(P4[i],L5,P4_L5[i]);
        //更新权重
        OP.iterate(PL);
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
void judge(const DataSet &testx, const DataSet &testy) {
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
    // train
    cout << "Training model" << endl;
    net::init();
    int split_position = 30000;
    judge(testx, testy);
    int T=500000;
    while(T--){
        int idx = randint(0, split_position - 1);
        net::train(trainx.data[idx], trainy.data[idx]);
        if(T%10000==0)judge(testx,testy);
    }
    judge(testx, testy);
    net::save("net_parameters.ini");
}
void train_network2(){
    net::load("net_parameters.ini");
    net2::init();
    int split_position = 30000;
    judge(testx, testy);
    int T=1000000;
    while(T--){
        int idx = randint(0, split_position - 1);
        net2::train(trainx.data[idx], trainy.data[idx]);
        if(T%10000==0)judge(testx,testy);
    }
    judge(testx, testy);
    net2::load("net2_parameters.ini");
}
void vis_network(Vector x,Vector y){
    net::init();
    net::load("net_parameters.ini");
    //image
    save_image(x,28,28,"image");
    //saliency_maps
    net::saliency_maps_back(x,y);
    save_image(net::I0.out_diff,28,28,"saliency_maps");
    //occlusion sensitivity
    function<Vector(Vector)> softmax=[](Vector x){
        double sum=0;
        for(auto &i:x)i=exp(i),sum+=i;
        for(auto &i:x)i=i/sum;
        return x;
    };
    int label=0;
    rep(i,0,9)if(y[i]>0.5)label=i;
    int d=7;
    Vector m((29-d)*(29-d),0);
    rep(i,0,28-d)rep(j,0,28-d){
        Vector xx=x;
        rep(k1,i,i+d-1)rep(k2,j,j+d-1)xx[k1*28+k2]=0.5;
        m[i*(29-d)+j]=softmax(net::predict(xx))[label];
    }
    save_image(m,29-d,29-d,"occlusion_sensitivity");
    //class activation map
    net2::load("net2_parameters.ini");
    string name="class_activation_map_0";
    rep(number,0,9){
        net2::predict(x);
        auto ans=net2::D3[0].out_val;
        ans.clear();
        rep(i,0,net2::num1)net2::D3[i].out_val*=net2::P4_L5[i].w[number];
        rep(i,0,net2::num1)ans+=net2::D3[i].out_val;
        save_image(ans,12,12,name.data());
        name[name.size()-1]++;
    }
}

int main() {
    read_data();
    //train_network();
    //train_network2();
    int idx=4;
    vis_network(trainx.data[idx],trainy.data[idx]);
    return 0;
}
/*

*///
