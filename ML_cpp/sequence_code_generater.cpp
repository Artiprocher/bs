#include "ML_Model.h"
#define rep(i,a,b) for(int i=(int)a;i<=(int)b;i++)
typedef long long ll;

void generate_code(vector<pair<string,string> > L,vector<pair<string,string> > E){
    set<string> edges;
    for(auto e:E)edges.insert(e.first+"_"+e.second);
    cout<<"namespace net{"<<endl;
    cout<<"    ParameterList PL;"<<endl;
    for(auto i:L)cout<<"    "<<i.first<<" "<<i.second<<";"<<endl;
    for(auto e:E)cout<<"    auto "<<e.first<<"_"<<e.second<<"=full_connect("<<e.first<<","<<e.second<<");"<<endl;
    cout<<"    function<Vector(Vector,Vector)> loss="<<endl;
    cout<<"    Optimazer::Adam OP;"<<endl;
    cout<<"    void init(){"<<endl;
    for(auto i:L)cout<<"        "<<i.second<<".get_parameters(PL);"<<endl;
    for(auto e:E)cout<<"        "<<e.first<<"_"<<e.second<<".get_parameters(PL);"<<endl;
    cout<<"        "<<"OP.init(PL);"<<endl;
    cout<<"    }"<<endl;
    cout<<"    Vector predict(const Vector &x){"<<endl;
    for(auto i:L)cout<<"        "<<i.second<<".clear();"<<endl;
    cout<<"        Vector2Array(x,"<<L[0].second<<".in_val);"<<endl;
    for(int i=0;i<(int)L.size();i++){
        if(i==0){
            ;
        }else if(edges.count(L[i-1].second+"_"+L[i].second)){
            cout<<"        "<<"push_forward("<<L[i-1].second<<","<<L[i].second<<","<<(L[i-1].second+"_"+L[i].second)<<");"<<endl;
        }else{
            cout<<"        "<<"push_forward("<<L[i-1].second<<","<<L[i].second<<");"<<endl;
        }
        cout<<"        "<<L[i].second<<".forward_solve();"<<endl;
    }
    cout<<"        return "<<L.back().second<<".out_val.Array2Vector();"<<endl;
    cout<<"    }"<<endl;
    cout<<"    void train(const Vector &x,const Vector &y){"<<endl;
    cout<<"        Vector y_=predict(x);"<<endl;
    cout<<"        Vector2Array(loss(y,y_),"<<L.back().second<<".in_diff);"<<endl;
    for(int i=(int)L.size()-1;i>=0;i--){
        if(i==(int)L.size()-1){
            ;
        }else if(edges.count(L[i].second+"_"+L[i+1].second)){
            cout<<"        "<<"push_backward("<<L[i].second<<","<<L[i+1].second<<","<<(L[i].second+"_"+L[i+1].second)<<");"<<endl;
        }else{
            cout<<"        "<<"push_backward("<<L[i].second<<","<<L[i+1].second<<");"<<endl;
        }
        cout<<"        "<<L[i].second<<".backward_solve();"<<endl;
    }
    cout<<"        OP.iterate(PL);"<<endl;
    cout<<"    }"<<endl;
    cout<<"}"<<endl;
}

int main(){
    generate_code(
        {
            {"DenseLayer<784>","IN"},
            {"ExpandParallel< ConvLayer<28,28,5,5>,6 >","C1"},
            {"Parallel< MaxPoolLayer<24,24,2,2>,6 >","S1"},
            {"Parallel< DenseLayer<12*12>,6 >","D1"},
            {"Parallel< ConvLayer<12,12,5,5>,16 >","C2"},
            {"Parallel< MaxPoolLayer<8,8,2,2>,16 >","S2"},
            {"Parallel< DenseLayer<4*4>,16 >","D2"},
            {"Parallel< ConvLayer<4,4,4,4>,16*100 >","C3"},
            {"DenseLayer<100>","D3"},
            {"DenseLayer<10>","OU"}
        },
        {
            {"C3","D3"},
            {"D3","OU"}
        }
    );
    return 0;
}
