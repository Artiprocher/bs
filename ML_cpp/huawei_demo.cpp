#define DEBUG
#include "ML_Model.h"

class SimpleNet{
public:
    ParameterList PL;
    DenseLayer<1000> IN;
    DenseLayer<128> H1=DenseLayer<128>(sigmoid,sigmoid_diff);
    DenseLayer<2> OU;
    ComplateEdge<1000,128> IN_H1=full_connect(IN,H1);
    ComplateEdge<128,2> H1_OU=full_connect(H1,OU);
    function<Vector(Vector,Vector)> loss=softmax_crossEntropy;
    Optimazer::BatchAdam<100> OP;
    void init(){
        OU.use_bias=1;
        IN.get_parameters(PL);
        H1.get_parameters(PL);
        OU.get_parameters(PL);
        IN_H1.get_parameters(PL);
        H1_OU.get_parameters(PL);
        OP.init(PL);
        cout<<"number of parameters: "<<PL.w.size()<<endl;
    }
    Vector predict(const Vector &x){
        IN.clear();
        H1.clear();
        OU.clear();
        Vector2Array(x,IN.in_val);
        IN.forward_solve();
        push_forward(IN,H1,IN_H1);
        H1.forward_solve();
        push_forward(H1,OU,H1_OU);
        OU.forward_solve();
        return OU.out_val.Array2Vector();
    }
    void train(const Vector &x,const Vector &y){
        Vector y_=predict(x);
        Vector2Array(loss(y,y_),OU.in_diff);
        OU.backward_solve();
        push_backward(H1,OU,H1_OU);
        H1.backward_solve();
        push_backward(IN,H1,IN_H1);
        IN.backward_solve();
        OP.iterate(PL);
    }
};

double loss(Vector y,Vector y_){
    double sum=exp(y_[0])+exp(y_[1]);
    for(auto &i:y_)i=exp(i)/sum;
    return -(y[0]*log(y_[0])+y[1]*log(y_[1]));
}

template <class NetClass> void judge(NetClass &net,const DataSet &x,const DataSet &y){
    int ac=0,all=x.data.size();
    Vector lo(1,0);
    for(int i=0;i<all;i++){
        Vector y_=net.predict(x.data[i]);
        int f1=y_[1]>y_[0],f2=y.data[i][1]>0.5;
        if(f1==f2)ac++;
        lo+=loss(y.data[i],y_);
    }
    cout<<"loss:"<<lo<<" ";
    cout<<"accuracy:"<<ac*1.0/all<<endl;
}

TXT_Reader txt_reader;
DataSet trainx,trainy,testx,testy;
SimpleNet net1;

void demo(){
    // read data
    cout << "Reading data" << endl;
    txt_reader.open("E:/hw/train.txt");
    txt_reader.export_number_data(0, 7200-1, 0, 999, trainx);
    txt_reader.export_onehot_data(0, 7200-1, 1000, trainy);
    txt_reader.close();
    txt_reader.open("E:/hw/val.txt");
    txt_reader.export_number_data(0, 800-1, 0, 999, testx);
    txt_reader.export_onehot_data(0, 800-1, 1000, testy);
    txt_reader.close();
    //train
    cout << "training model" << endl;
    net1.init();
    int epoch=10000;
    judge(net1,testx,testy);
    while(1){
        for(int it=1;it<=epoch;it++){
            int idx=randint(0,7200-1);
            net1.train(trainx.data[idx],trainy.data[idx]);
        }
        judge(net1,testx,testy);
    }
}

int main() {
    demo();
    return 0;
}
/*

*///
