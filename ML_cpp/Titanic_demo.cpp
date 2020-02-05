#include "ML_Model.h"

namespace net3{
    double eta=0.5;
    ActiveLayer<25> input_layer;
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
        hidden_layer.forward_solve();
        push_forward(hidden_layer,output_layer,E2);
        output_layer.forward_solve();
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
DataSet x,y,t;

void judge3() {
    int l=601,r=890,all = r-l+1, ac = 0;
    rep(it, l, r) {
        Vector a = net3::predict(x.data[it]);
        if(fabs(a[0]-y.data[it][0])<0.5)ac++;
    }
    cout << (ac * 1.0 / all) << endl;
}

void demo3(){
    cout << "Reading data" << endl;
    csv_reader.open("Titanic/train.csv");
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
    x+=t;
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
    demo3();
    return 0;
}
/*

*///
