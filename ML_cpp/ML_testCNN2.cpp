#include "ML_test.h"

namespace net1{
    double eta=0.05;
    ActiveLayer<784> input_layer;
    ActiveLayer<100> hidden_layer(sigmoid,sigmoid_diff);
    ActiveLayer<10> output_layer(sigmoid,sigmoid_diff);
    auto E1=full_connect(input_layer,hidden_layer);
    auto E2=full_connect(hidden_layer,output_layer);
    auto loss=crossEntropy;
    void init(){
        ;
    }
    Vector predict(const Vector &x){
        /*清理*/
        input_layer.clear();
        hidden_layer.clear();
        output_layer.clear();
        /*正向传值*/
        each_index(i,x)input_layer.out_val[i]=x[i];
        push_forward(input_layer,hidden_layer,E1);
        push_forward(hidden_layer,output_layer,E2);
        /*导出结果*/
        static Vector y(output_layer.output_size,0);
        for(int i=0;i<output_layer.output_size;i++){
            y[i]=output_layer.out_val[i];
        }
        return y;
    }
    void train(const Vector &x,const Vector &y){
        /*正向传值*/
        Vector y_=predict(x);
        /*逆向传值*/
        Vector2Array(loss(y,y_),output_layer.diff_val);
        push_backward(hidden_layer,output_layer,E2,eta);
        push_backward(input_layer,hidden_layer,E1,eta);
        /*更新权重*/
        hidden_layer.update_w(eta);
        output_layer.update_w(eta);
    }
};

namespace net2{
    const int num1=6;
    double eta=0.05;
    ActiveLayer<784> I0;
    ConvLayer<28,28,5,5> C1[num1];
    MaxPoolLayer<24,24,2,2> S2[num1];
    ActiveLayer<10> L3(sigmoid,sigmoid_diff);
    ComplateEdge<144,10> E3[num1];
    auto loss=crossEntropy;
    void init(){
        //cout<<fixed<<setprecision(2);
        for(int i=0;i<num1;i++)S2[i]=MaxPoolLayer<24,24,2,2>(sigmoid,sigmoid_diff);
        for(int i=0;i<num1;i++)E3[i]=full_connect(S2[i],L3);
    }
    Vector predict(const Vector &x){
        /*清理*/
        I0.clear();
        for(int i=0;i<num1;i++)C1[i].clear();
        for(int i=0;i<num1;i++)S2[i].clear();
        L3.clear();
        /*正向传值*/
        each_index(i,x)I0.out_val[i]=x[i];
        for(int i=0;i<num1;i++)push_forward(I0,C1[i]);
        for(int i=0;i<num1;i++)C1[i].forward_solve();
        for(int i=0;i<num1;i++)push_forward(C1[i],S2[i]);
        for(int i=0;i<num1;i++)push_forward(S2[i],L3,E3[i]);
        /*导出结果*/
        static Vector y(L3.output_size,0);
        for(int i=0;i<L3.output_size;i++){
            y[i]=L3.out_val[i];
        }
        return y;
    }
    void train(const Vector &x,const Vector &y){
        /*正向传值*/
        Vector y_=predict(x);
        /*逆向传值*/
        Vector2Array(loss(y,y_),L3.diff_val);
        for(int i=0;i<num1;i++)push_backward(S2[i],L3,E3[i],eta);
        for(int i=0;i<num1;i++)push_backward(C1[i],S2[i],eta);
        for(int i=0;i<num1;i++)push_backward(I0,C1[i],eta);
        /*更新权重*/
        I0.update_w(eta);
        for(int i=0;i<num1;i++)C1[i].update_w(eta);
        for(int i=0;i<num1;i++)S2[i].update_w(eta);
        L3.update_w(eta);
    }
}

namespace net3{
    double eta=0.5;
    ActiveLayer<42> input_layer;
    ActiveLayer<20> hidden_layer(sigmoid,sigmoid_diff);
    ActiveLayer<10> output_layer(sigmoid,sigmoid_diff);
    auto E1=full_connect(input_layer,hidden_layer);
    auto E2=full_connect(hidden_layer,output_layer);
    auto loss=mse;
    void init(){
        ;
    }
    Vector predict(const Vector &x){
        /*清理*/
        input_layer.clear();
        hidden_layer.clear();
        output_layer.clear();
        /*正向传值*/
        each_index(i,x)input_layer.out_val[i]=x[i];
        push_forward(input_layer,hidden_layer,E1);
        push_forward(hidden_layer,output_layer,E2);
        /*导出结果*/
        static Vector y(output_layer.output_size,0);
        for(int i=0;i<output_layer.output_size;i++){
            y[i]=output_layer.out_val[i];
        }
        return y;
    }
    void train(const Vector &x,const Vector &y){
        /*正向传值*/
        Vector y_=predict(x);
        /*逆向传值*/
        Vector2Array(loss(y,y_),output_layer.diff_val);
        push_backward(hidden_layer,output_layer,E2,eta);
        push_backward(input_layer,hidden_layer,E1,eta);
        /*更新权重*/
        hidden_layer.update_w(eta);
        output_layer.update_w(eta);
    }
};

CSV_Reader csv_reader;
DataSet trainx,trainy,testx,testy,x,y,t;

void show_image(const Vector &a){
    rep(i,0,783){
        cout<<(a[i]>0.5?"*":" ");
        if((i+1)%28==0)cout<<endl;
    }
    cout<<endl;
}
void judge1(const DataSet &testx, const DataSet &testy) {
    int all = testx.data.size(), ac = 0;
    rep(it, 0, all - 1) {
        Vector a = net1::predict(testx.data[it]);
        int ans = 0;
        rep(i, 1, 9) {
            if (a[i] > a[ans]) ans = i;
        }
        if (testy.data[it][ans] > 0.5) ac++;
    }
    cout << (ac * 1.0 / all) << endl;
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
void judge3() {
    int l=601,r=890,all = r-l+1, ac = 0;
    rep(it, l, r) {
        Vector a = net3::predict(x.data[it]);
        if(fabs(a[0]-y.data[it][0])<0.5)ac++;
    }
    cout << (ac * 1.0 / all) << endl;
}
void demo1(){
    // read data
    cout << "Reading data" << endl;
    csv_reader.open("train.csv");
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
    net1::init();
    judge1(testx, testy);
    ll epoch = 1000000, goal = 1;
    rep(it, 1, epoch) {
        int idx = randint(0, split_position - 1);
        net1::train(trainx.data[idx], trainy.data[idx]);
        if (it * 100 >= epoch * goal) {
            cout << it * 100.0 / epoch << "%" << endl;
            if (goal % 10 == 0) {
                cout << "accuracy:";
                judge1(testx, testy);
            }
            goal++;
        }
    }
}
void demo2(){
    // read data
    cout << "Reading data" << endl;
    csv_reader.open("train.csv");
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
    ll epoch = 100000, goal = 1;
    rep(it, 1, epoch) {
        int idx = randint(0, split_position - 1);
        net2::train(trainx.data[idx], trainy.data[idx]);
        if (it * 100 >= epoch * goal) {
            cout << it * 100.0 / epoch << "%" << endl;
            if (goal % 10 == 0) {
                cout << "accuracy:";
                judge2(testx, testy);
            }
            goal++;
        }
    }
}
void demo3(){
    // read data
    cout << "Reading data" << endl;
    csv_reader.open("Titanic/train.csv");
    //csv_reader.describe();
    csv_reader.shuffle();
    //Pclass
    csv_reader.export_onehot_data(0, 890, 2, x);
    //Sex
    csv_reader.export_onehot_data(0, 890, 4, t);
    x+=t;
    //Age
    csv_reader.export_number_data(0, 890, 5, 5, t);
    t.fill_nan_with_mean();
    t.min_max_normalization(0);
    x+=t;
    //SibSp
    csv_reader.export_onehot_data(0, 890, 6, t);
    x+=t;
    //Parch
    csv_reader.export_onehot_data(0, 890, 7, t);
    x+=t;
    //Fare
    csv_reader.export_number_data(0, 890, 9, 9, t);
    t.fill_nan_with_mean();
    t.min_max_normalization(0);
    //Embarked
    csv_reader.export_onehot_data(0, 890, 11, t);
    x+=t;
    //Survived
    csv_reader.export_number_data(0, 890, 1, 1, y);

    cout << "Training model" << endl;
    net3::init();
    ll epoch = 1000000, goal = 1;
    rep(it, 1, epoch) {
        int idx = randint(0, 600);
        net3::train(x.data[idx], y.data[idx]);
        if (it * 100 >= epoch * goal) {
            cout << it * 100.0 / epoch << "%" << endl;
            if (goal % 10 == 0) {
                cout << "accuracy:";
                judge3();
            }
            goal++;
        }
    }
}

int main() {
    demo2();
    return 0;
}
/*

*///
