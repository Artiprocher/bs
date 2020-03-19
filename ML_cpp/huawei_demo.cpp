#define DEBUG
#include "ML_Model.h"

class SimpleNet{
public:
    ParameterList PL;
    DenseLayer<1000>  IN;
    DenseLayer<128>   H1=DenseLayer<128>(sigmoid,sigmoid_diff);
    DenseLayer<128>   H2=DenseLayer<128>(sigmoid,sigmoid_diff);
    DropoutLayer<128> D3=DropoutLayer<128>(0);
    DenseLayer<1>     OU=DenseLayer<1>(sigmoid,sigmoid_diff);
    ComplateEdge<1000,128> IN_H1=full_connect(IN,H1);
    ComplateEdge<128,128>  H1_H2=full_connect(H1,H2);
    ComplateEdge<128,1>    D3_OU=full_connect(D3,OU);
    function<Vector(Vector,Vector)> loss=crossEntropy;
    Optimazer::BatchAdam<100> OP;
    void init(){
        IN.get_parameters(PL);
        H1.get_parameters(PL);
        H2.get_parameters(PL);
        D3.get_parameters(PL);
        OU.get_parameters(PL);
        IN_H1.get_parameters(PL);
        H1_H2.get_parameters(PL);
        D3_OU.get_parameters(PL);
        OP.init(PL);
        cout<<"number of parameters: "<<PL.w.size()<<endl;
    }
    void load(const char *file_name) {
        ifstream f;
        f.open(file_name,ios::in);
        PL.load(f);
        f.close();
    }
    void load(stringstream &f) {
        for(auto &i:PL.w)f>>(*i);
    }
    void save(const char *file_name) {
        ofstream f;
        f.open(file_name,ios::out);
        PL.save(f);
        f.close();
    }
    Vector predict(const Vector &x){
        IN.clear();
        H1.clear();
        H2.clear();
        D3.clear();
        OU.clear();
        Vector2Array(x,IN.in_val);
        IN.forward_solve();
        push_forward(IN,H1,IN_H1);
        H1.forward_solve();
        push_forward(H1,H2,H1_H2);
        H2.forward_solve();
        push_forward(H2,D3);
        D3.forward_solve();
        push_forward(D3,OU,D3_OU);
        OU.forward_solve();
        return OU.out_val.Array2Vector();
    }
    void train(const Vector &x,const Vector &y){
        D3.set_drop_probability(0.5);
        Vector y_=predict(x);
        Vector2Array(loss(y,y_),OU.in_diff);
        OU.backward_solve();
        push_backward(D3,OU,D3_OU);
        D3.backward_solve();
        push_backward(H2,D3);
        H2.backward_solve();
        push_backward(H1,H2,H1_H2);
        H1.backward_solve();
        push_backward(IN,H1,IN_H1);
        IN.backward_solve();
        OP.iterate(PL);
        D3.set_drop_probability(0.0);
    }
};

/*class FuckNet{
public:
    ParameterList PL;
    DenseLayer<1000> IN;
    DenseLayer<1> OU=DenseLayer<1>(sigmoid,sigmoid_diff);
    ComplateEdge<1000,1> IN_OU=full_connect(IN,OU);
    function<Vector(Vector,Vector)> loss=crossEntropy;
    Optimazer::BatchAdam<100> OP;
    void init(){
        IN.get_parameters(PL);
        OU.get_parameters(PL);
        IN_OU.get_parameters(PL);
        OP.init(PL);
        cout<<"number of parameters: "<<PL.w.size()<<endl;
    }
    void load(const char *file_name) {
        ifstream f;
        f.open(file_name,ios::in);
        PL.load(f);
        f.close();
    }
    void load(stringstream &f) {
        for(auto &i:PL.w)f>>(*i);
    }
    void save(const char *file_name) {
        ofstream f;
        f.open(file_name,ios::out);
        PL.save(f);
        f.close();
    }
    Vector predict(const Vector &x){
        IN.clear();
        OU.clear();
        Vector2Array(x,IN.in_val);
        IN.forward_solve();
        push_forward(IN,OU,IN_OU);
        OU.forward_solve();
        return OU.out_val.Array2Vector();
    }
    void train(const Vector &x,const Vector &y){
        Vector y_=predict(x);
        Vector2Array(loss(y,y_),OU.in_diff);
        OU.backward_solve();
        push_backward(IN,OU,IN_OU);
        IN.backward_solve();
        OP.iterate(PL);
    }
};*/

template <class NetClass> void judge(NetClass &net,const DataSet &x,const DataSet &y){
    int ac=0,all=x.data.size();
    for(int i=0;i<all;i++){
        Vector y_=net.predict(x.data[i]);
        int f1=y_[0]<0.5,f2=y.data[i][0]<0.5;
        if(f1==f2)ac++;
    }
    cout<<"accuracy:"<<ac*1.0/all<<endl;
}
TXT_Reader txt_reader;
DataSet trainx,trainy,testx,testy;
SimpleNet net1;
vector<Vector> one,zero;

void demo(){
    // read data
    cout << "Reading data" << endl;
    txt_reader.open("E:/hw/data/train_data.txt");
    txt_reader.export_number_data(0, txt_reader.data.size()-1, 0, 999, trainx);
    txt_reader.export_number_data(0, txt_reader.data.size()-1, 1000, 1000, trainy);
    txt_reader.close();
    txt_reader.open("E:/hw/data/test_data.txt");
    txt_reader.export_number_data(0, txt_reader.data.size()-1, 0, 999, testx);
    txt_reader.close();
    txt_reader.open("E:/hw/data/answer.txt");
    txt_reader.export_number_data(0, txt_reader.data.size()-1, 0, 0, testy);
    txt_reader.close();
    //train
    cout << "training model" << endl;
    net1.init();
    for(int i=0;i<(int)trainx.data.size();i++){
        if(trainy.data[i][0]<0.5)zero.emplace_back(trainx.data[i]);
        else one.emplace_back(trainx.data[i]);
    }
    cout<<zero.size()<<" "<<one.size()<<endl;
    //net1.load("huawei_fuck_fuck_fuck.ini");
    judge(net1,testx,testy);
    int epoch=10000;
    for(int t=1;t<=10000;t++){
        for(int it=1;it<=epoch;it++){
            int idx=randint(0,trainx.data.size()-1);
            net1.train(trainx.data[idx],trainy.data[idx]);
        }
        cout<<"train: ";
        judge(net1,trainx,trainy);
        cout<<"test: ";
        judge(net1,testx,testy);
        net1.save("huawei_fuck_fuck.ini");
    }
}

int main() {
    demo();
    return 0;
}
/*

*///
