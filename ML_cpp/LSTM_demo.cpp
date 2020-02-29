#include "ML_Model.h"
#define rep(i,a,b) for(int i=(int)a;i<=(int)b;i++)
typedef long long ll;

/*namespace SimpleRNN{
    const int n=30,m=20,k=1;
    const int timestep=40;
    const double eta=0.05,eta_=eta;
    auto loss=mse;
    class cell{
    public:
        DenseLayer<n> X;
        DenseLayer<m> H;
        DenseLayer<k> Y;
        ComplateEdge<n,m> X_H=full_connect(X,H);
        ComplateEdge<m,k> H_Y=full_connect(H,Y);
        ComplateEdge<m,m> H_H=full_connect(H,H);
        cell(){
            H=DenseLayer<m>(sigmoid,sigmoid_diff);
            Y=DenseLayer<k>(constant,constant_diff);
        }
        void clear(){
            X.clear();
            H.clear();
            Y.clear();
        }
        void show(){
            cout<<"  X_H"<<endl;
            X_H.show();
            cout<<"  H_Y"<<endl;
            H_Y.show();
        }
        void forward_disseminate(){
            X.forward_solve();
            push_forward(X,H,X_H);
            H.forward_solve();
            push_forward(H,Y,H_Y);
            Y.forward_solve();
        }
        void backward_disseminate(){
            Y.backward_solve();
            push_backward(H,Y,H_Y,eta_);
            H.backward_solve();
            push_backward(X,H,X_H,eta_);
        }
        void update_w(){
            X.update_w(eta_);
            H.update_w(eta_);
            Y.update_w(eta_);
        }
    }c[timestep];
    void init(){}
    vector<Vector> predict(const vector<Vector> &vx){
        assert(vx.size()==timestep);
        for(int i=0;i<timestep;i++)c[i].clear();
        for(int i=0;i<timestep;i++)Vector2Array(vx[i],c[i].X.in_val);
        for(int i=0;i<timestep;i++){
            if(i>0)push_forward(c[i-1].H,c[i].H,c[i-1].H_H);
            c[i].forward_disseminate();
        }
        vector<Vector> vy;
        for(int i=0;i<timestep;i++)vy.emplace_back(Array2Vector(c[i].Y.out_val));
        return vy;
    }
    void train(const vector<Vector> &vx,const vector<Vector> &vy){
        assert(vx.size()==timestep && vy.size()==timestep);
        vector<Vector> y_=predict(vx);
        //沿时间轴逆向传播
        for(int i=0;i<timestep;i++)Vector2Array(loss(vy[i],y_[i]),c[i].Y.in_diff);
        for(int i=timestep-1;i>=0;i--){
            c[i].backward_disseminate();
            if(i>0)push_backward(c[i-1].H,c[i].H,c[i-1].H_H,eta_);
        }
        //更新权重
        for(int i=0;i<timestep;i++)c[i].update_w();
        for(int i=1;i<timestep;i++){
            c[0].H.c+=c[i].H.c;c[0].Y.c+=c[i].Y.c;
            c[0].X_H+=c[i].X_H;c[0].H_Y+=c[i].H_Y;
            if(i!=timestep-1)c[0].H_H+=c[i].H_H;
        }
        static const double temp=1.0/timestep;
        c[0].H.c*=temp;
        c[0].Y.c*=temp;
        c[0].X_H*=temp;
        c[0].H_Y*=temp;
        c[0].H_H*=1.0/(timestep-1);
        for(int i=1;i<timestep;i++)c[i]=c[0];
    }
}*/

namespace LSTM{
    ParameterList PL;
    const int n=30,m=20,k=1;
    const int timestep=40;
    const double eta=0.05;
    auto loss=mse;
    Optimazer::GradientDescent GD(eta);
    class cell{
    public:
        DenseLayer<n> X;
        DenseLayer<m> H0,D[4],C0,C2,C3;
        DenseLayer<k> Y;
        MultiplicationLayer<m> C1,H1,M;
        ComplateEdge<n,m> X_D[4];
        ComplateEdge<m,m> H_D[4];
        ComplateEdge<m,k> H1_Y;
        cell(){
            D[0]=DenseLayer<m>(sigmoid,sigmoid_diff);
            D[1]=DenseLayer<m>(sigmoid,sigmoid_diff);
            D[2]=DenseLayer<m>(Tanh,Tanh_diff);
            D[3]=DenseLayer<m>(sigmoid,sigmoid_diff);
            Y=DenseLayer<k>(constant,constant_diff);
            C3=DenseLayer<m>(Tanh,Tanh_diff);
            X.get_parameters(PL);H0.get_parameters(PL);
            D[0].get_parameters(PL);D[1].get_parameters(PL);D[2].get_parameters(PL);D[3].get_parameters(PL);
            M.get_parameters(PL);
            C0.get_parameters(PL);C1.get_parameters(PL);C2.get_parameters(PL);C3.get_parameters(PL);
            H1.get_parameters(PL);Y.get_parameters(PL);
            for(int i=0;i<4;i++)X_D[i].get_parameters(PL);
            for(int i=0;i<4;i++)H_D[i].get_parameters(PL);
            H1_Y.get_parameters(PL);
        }
        void clear(){
            X.clear();H0.clear();
            D[0].clear();D[1].clear();D[2].clear();D[3].clear();
            M.clear();
            C0.clear();C1.clear();C2.clear();C3.clear();
            H1.clear();Y.clear();
        }
        void forward_disseminate(){
            X.forward_solve();
            H0.forward_solve();
            for(int i=0;i<4;i++){
                push_forward(X,D[i],X_D[i]);
                push_forward(H0,D[i],H_D[i]);
                D[i].forward_solve();
            }
            C0.forward_solve();
            push_forward(C0,D[0],C1);C1.forward_solve();
            push_forward(D[1],D[2],M);M.forward_solve();
            push_forward(C1,C2);push_forward(M,C2);C2.forward_solve();
            push_forward(C2,C3);C3.forward_solve();
            push_forward(C3,D[3],H1);H1.forward_solve();
            push_forward(H1,Y,H1_Y);Y.forward_solve();
        }
        void backward_disseminate(){
            Y.backward_solve();push_backward(H1,Y,H1_Y);
            H1.backward_solve();push_backward(C3,D[3],H1);
            C3.backward_solve();push_backward(C2,C3);
            C2.backward_solve();push_backward(C1,C2);push_backward(M,C2);
            C1.backward_solve();push_backward(C0,D[0],C1);
            M.backward_solve();push_backward(D[1],D[2],M);
            for(int i=0;i<4;i++){
                D[i].backward_solve();
                push_backward(H0,D[i],H_D[i]);
                push_backward(X,D[i],X_D[i]);
            }
        }
    }c[timestep];
    void init(){}
    vector<Vector> predict(const vector<Vector> &vx){
        assert(vx[0].size()==n);
        assert(vx.size()==timestep);
        for(int i=0;i<timestep;i++)c[i].clear();
        for(int i=0;i<timestep;i++)Vector2Array(vx[i],c[i].X.in_val);
        for(int i=0;i<timestep;i++){
            if(i>0){
                push_forward(c[i-1].H1,c[i].H0);
                push_forward(c[i-1].C2,c[i].C0);
            }
            c[i].forward_disseminate();
        }
        vector<Vector> vy;
        for(int i=0;i<timestep;i++)vy.emplace_back(Array2Vector(c[i].Y.out_val));
        return vy;
    }
    void train(const vector<Vector> &vx,const vector<Vector> &vy){
        assert(vx.size()==timestep && vy.size()==timestep);
        assert(vx[0].size()==n && vy[0].size()==k);
        vector<Vector> y_=predict(vx);
        //沿时间轴逆向传播
        for(int i=0;i<timestep;i++)Vector2Array(loss(vy[i],y_[i]),c[i].Y.in_diff);
        for(int i=timestep-1;i>=0;i--){
            c[i].backward_disseminate();
            if(i>0){
                push_backward(c[i-1].H1,c[i].H0);
                push_backward(c[i-1].C2,c[i].C0);
            }
        }
        //更新权重
        GD.iterate(PL);
        const double temp=1.0/timestep;
        int num=PL.w.size()/timestep;
        for(int t=1;t<timestep;t++){
            for(int i=0;i<num;i++)*PL.w[i]+=*PL.w[t*num+i];
        }
        for(int i=0;i<num;i++)*PL.w[i]*=temp;
        for(int t=1;t<timestep;t++){
            for(int i=0;i<num;i++)*PL.w[t*num+i]=*PL.w[i];
        }
    }
}

#define NET LSTM
CSV_Reader csv_reader;

namespace SINGLE_LSTM_DEMO{
    DataSet data;
    double f(int x){
        return data(x,0);
    }
    vector<Vector> get_vx(int l,int r){
        vector<Vector> vx(NET::timestep,Vector(NET::n,0));
        for(int i=l;i<=r;i++){
            for(int j=0;j<NET::n;j++)vx[i-l][j]=f(i-NET::n+j);
        }
        return vx;
    }
    vector<Vector> get_vy(int l,int r){
        vector<Vector> vy(NET::timestep,Vector(NET::k,0));
        for(int i=l;i<=r;i++)vy[i-l]=(Vector){f(i)};
        return vy;
    }
    void demo(){
        //read data
        csv_reader.open("weather/beijing_data.csv");
        int all=csv_reader.size()[0];
        csv_reader.export_number_data(0,all-1,7,7,data);
        cout<<fixed<<setprecision(6)<<data.mean(0)<<" "<<data.std_dev(0)<<endl;
        data.zscore_normalization(0);
        //train
        int train=34000,test=2000,epoch=200000;
        for(int it=1;it<=epoch;it++){
            int l=randint(NET::n,train-NET::timestep),r=l+NET::timestep-1;
            NET::train(get_vx(l,r),get_vy(l,r));
            if(it%(epoch/100)==0)cout<<it*100.0/epoch<<"%"<<endl;
        }
        //test
        ofstream fout;
        fout.open("weather/predict.csv",ios::out);
        double loss=0,baseline_loss=0;
        for(int i=train+1;i<=train+test;i++){
            double y=f(i),y_=NET::predict(get_vx(i-NET::timestep+1,i)).back()[0];
            fout<<fixed<<setprecision(6)<<y_<<endl;
            loss+=abs(y-y_);
            baseline_loss+=abs(y-f(i-1));
            data(i,0)=y_;
        }
        cout<<"loss="<<loss<<endl;
        cout<<"baseline_loss="<<baseline_loss<<endl;
    }
}

int main() {
    SINGLE_LSTM_DEMO::demo();
    return 0;
}
/*

*///
