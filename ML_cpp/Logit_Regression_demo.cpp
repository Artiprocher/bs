#include <bits/stdc++.h>
#define rep(i,a,b) for(int i=(a);i<=(int)(b);i++)
#include "ML_Model.h"
using namespace std;

CSV_Reader csv_reader;
LogitRegression L[10];
DataSet trainx,trainy,testx,testy;

void show_image(const Vector &a){
    rep(i,0,783){
        cout<<(a[i]>0.5?"*":" ");
        if((i+1)%28==0)cout<<endl;
    }
    cout<<endl;
}
void judge(const DataSet &testx,const DataSet &testy){
    int all=testx.data.size(),ac=0;
    rep(it,0,all-1){
        int ans=0;
        rep(i,1,9){
            if(L[i].predict(testx.data[it])>L[ans].predict(testx.data[it])){
                ans=i;
            }
        }
        if(testy.data[it][ans]>0.5)ac++;
    }
    std::cout<<(ac*1.0/all)<<std::endl;
}

int main() {
    //read data
    cout<<"Reading data"<<endl;
    csv_reader.open("train.csv");
    csv_reader.shuffle();
    int split_position=30000;
    csv_reader.export_number_data(1,split_position,1,784,trainx);
    csv_reader.export_onehot_data(1,split_position,0,trainy);
    csv_reader.export_number_data(split_position+1,42000,1,784,testx);
    csv_reader.export_onehot_data(split_position+1,42000,0,testy);
    csv_reader.close();
    rep(i,0,trainx.data.size()-1)trainx.data[i]*=1.0/255;
    rep(i,0,testx.data.size()-1)testx.data[i]*=1.0/255;
    //model init
    rep(i,0,9){
        L[i].eta=0.1;
        L[i].init(784);
    }
    //train
    cout<<"Training model"<<endl;
    int epoch=1000000;
    rep(it,1,epoch){
        int idx=randint(0,split_position-1);
        rep(i,0,9)L[i].train(trainx.data[idx],trainy.data[idx][i]);
        if(it%10000==0)cout<<it/10000<<"%"<<endl;
    }
    L[0].save("L0.ini");
    L[1].save("L1.ini");
    L[2].save("L2.ini");
    L[3].save("L3.ini");
    L[4].save("L4.ini");
    L[5].save("L5.ini");
    L[6].save("L6.ini");
    L[7].save("L7.ini");
    L[8].save("L8.ini");
    L[9].save("L9.ini");
    //judge
    cout<<"Judging model"<<endl;
    judge(testx,testy);
    return 0;
}

