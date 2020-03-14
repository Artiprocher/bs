#define DEBUG
#include <bits/stdc++.h>
using namespace std;
void quitf(std::string mes){
    std::cerr<<mes<<std::endl;
    exit(0);
}

#ifndef ML_Vector
#define ML_Vector

#include <bits/stdc++.h>
using namespace std;
#include "ML_Rand.h"
typedef std::vector<double> Vector;
#define each_index(i,a) for(int i=0;i<(int)a.size();i++)
Vector operator += (Vector &a,const Vector &b){each_index(i,a)a[i]+=b[i];return a;}
Vector operator -= (Vector &a,const Vector &b){each_index(i,a)a[i]-=b[i];return a;}
Vector operator *= (Vector &a,const Vector &b){each_index(i,a)a[i]*=b[i];return a;}
Vector operator /= (Vector &a,const Vector &b){each_index(i,a)a[i]/=b[i];return a;}
Vector operator + (const Vector &a,const Vector &b){Vector c=a;return c+=b;}
Vector operator - (const Vector &a,const Vector &b){Vector c=a;return c-=b;}
Vector operator * (const Vector &a,const Vector &b){Vector c=a;return c*=b;}
Vector operator / (const Vector &a,const Vector &b){Vector c=a;return c/=b;}
Vector operator += (Vector &a,const double &b){each_index(i,a)a[i]+=b;return a;}
Vector operator -= (Vector &a,const double &b){each_index(i,a)a[i]-=b;return a;}
Vector operator *= (Vector &a,const double &b){each_index(i,a)a[i]*=b;return a;}
Vector operator /= (Vector &a,const double &b){each_index(i,a)a[i]/=b;return a;}
Vector operator + (const Vector &a,const double b){Vector c=a;return c+=b;}
Vector operator - (const Vector &a,const double b){Vector c=a;return c-=b;}
Vector operator * (const Vector &a,const double b){Vector c=a;return c*=b;}
Vector operator / (const Vector &a,const double b){Vector c=a;return c/=b;}
Vector operator + (const double b,const Vector &a){Vector c=a;return c+=b;}
Vector operator - (const double b,const Vector &a){return Vector(a.size(),b)-a;}
Vector operator * (const double b,const Vector &a){Vector c=a;return c*=b;}
Vector operator / (const double b,const Vector &a){return Vector(a.size(),b)/a;}
Vector operator - (const Vector &a){Vector c=a;each_index(i,c)c[i]=-c[i];return c;}
double Dot(const Vector &a,const Vector &b){
    assert(a.size()==b.size());
    double ans=0;
    each_index(i,a)ans+=a[i]*b[i];
    return ans;
}
Vector connect(const Vector &a,const Vector &b){Vector c=a;c.insert(c.end(),b.begin(),b.end());return c;}
double sum(const Vector &a){return std::accumulate(a.begin(),a.end(),0.0);}
std::istream &operator >>(std::istream &in,Vector &a){double x;in>>x;a.push_back(x);return in;}
std::ostream &operator <<(std::ostream &out,const Vector &a){
    out<<'(';
    each_index(i,a){
        out<<a[i];
        if(i+1!=(int)a.size())out<<',';
    }
    out<<')';
    return out;
}
template <class fun> void each(Vector &a,fun op){
    each_index(i,a)op(a[i]);
}
template <class ArrayType>void Vector2Array(const Vector &a,ArrayType &b){
    each_index(i,a)b[i]=a[i];
}

class Tensor{
private:
    std::vector<double> data;
    std::vector<int> siz;
public:
    void resize(const std::vector<int> &s){
        siz=s;
        int all=1;
        for(auto i:siz)all*=i;
        data.resize(all);
    }
    Tensor(const std::vector<int> &s){
        resize(s);
    }
    std::vector<int> size()const{
        return siz;
    }
#ifndef DEBUG
    double &operator () (int a,...){
        int index=a;
        va_list arg_ptr;
        va_start(arg_ptr,a);
        for(int i=1;i<(int)siz.size();i++){
            index=index*siz[i]+va_arg(arg_ptr,int);
        }
        return data[index];
    }
#else
    double &operator () (int a,...){
        int index=a;
        assert(a>=0 && a<siz[0]);
        va_list arg_ptr;
        va_start(arg_ptr,a);
        for(int i=1;i<(int)siz.size();i++){
            int p=va_arg(arg_ptr,int);
            assert(p>=0 && p<siz[i]);
            index=index*siz[i]+p;
        }
        assert(index>=0 && index<(int)data.size());
        return data[index];
    }
#endif
};

//智能数组 用[]使用一维索引 用()使用二维索引
template <const int N,const int M>
class SmartArray{
public:
    double data[N*M];
    void clear(){for(int i=0;i<N*M;i++)data[i]=0;}
    SmartArray<N,M>(){clear();}
    void show(){for(int i=0;i<N;i++)for(int j=0;j<M;j++)std::cout<<data[i*M+j]<<",\n"[j+1==M];}
    double& operator [] (int x){return data[x];}
    double& operator () (int x,int y){return data[x*M+y];}
    void reset_weight(double l,double r){for(int i=0;i<N*M;i++)data[i]=Rand(l,r);}
    void operator += (const SmartArray<N,M> &b){for(int i=0;i<N*M;i++)data[i]+=b.data[i];}
    void operator *= (const SmartArray<N,M> &b){for(int i=0;i<N*M;i++)data[i]*=b.data[i];}
    void operator += (const double &b){for(int i=0;i<N*M;i++)data[i]+=b;}
    void operator *= (const double &b){for(int i=0;i<N*M;i++)data[i]*=b;}
    Vector Array2Vector()const{return Vector(data,data+N*M);}
};
template <const int N,const int M>
Vector Array2Vector(const SmartArray<N,M> &a){return Vector(a.data,a.data+N*M);}

#endif


#ifndef ML_Rand
#define ML_Rand

#include <bits/stdc++.h>
using namespace std;

#ifdef DEBUG
double Rand(double l=0,double r=1){
    static default_random_engine generator(0);
    static uniform_real_distribution<double> rd(0.0,1.0);
    return rd(generator)*(r-l)+l;
}
int randint(int l,int r){
    static mt19937 rd(0);
    return rd()%(r-l+1)+l;
}
double NormalRand(double mu=0.0,double sigma=1.0){
    static default_random_engine generator(0);
    static normal_distribution<double> rd(0.0,1.0);
    return rd(generator)*sigma+mu;
}
#else
double Rand(double l=0,double r=1){
    static default_random_engine generator(time(0));
    static uniform_real_distribution<double> rd(0.0,1.0);
    return rd(generator)*(r-l)+l;
}
int randint(int l,int r){
    static mt19937 rd(time(0));
    return rd()%(r-l+1)+l;
}
double NormalRand(double mu=0.0,double sigma=1.0){
    static default_random_engine generator(time(0));
    static normal_distribution<double> rd(0.0,1.0);
    return rd(generator)*sigma+mu;
}
#endif

#endif

#ifndef ML_Linear_Model
#define ML_Linear_Model

#include <bits/stdc++.h>
using namespace std;
#include "ML_Rand.h"
#include "ML_Vector.h"

class LinearRegression {
   private:
    Vector w;

   public:
    double eta = 0.1;
    void init(int n) {
        w.resize(n);
        for (auto &i : w) i = Rand() - 0.5;
    }
    LinearRegression() {}
    LinearRegression(int n) { init(n); }
    void show() const {
        cout << " y =";
        each_index(i, w) {
            cout << " "[i == 0] << "(" << w[i] << ") * x" << i << " "
                      << "+\n"[i + 1 == (int)w.size()];
        }
    }
    double predict(const Vector &x) const {
        assert(x.size() == w.size());
        return Dot(x, w);
    }
    void train(const Vector &x, const double &y) {
        assert(x.size() == w.size());
        double y_ = predict(x);
        w -= (eta * (y_ - y)) * x;
        for(auto i:w)if(isinf(i)){
            cerr<<"Divergence!"<<endl;
            exit(1);
        }
    }
    void save(const char *file_name) const {
        ofstream fout;
        fout.open(file_name, ios::out);
        fout << "LinearRegression" << endl;
        for (auto i : w) {
            fout << fixed << setprecision(10) << i << endl;
        }
    }
    void load(const char *file_name) {
        ifstream fin;
        fin.open(file_name, ios::in);
        w.clear();
        string model_name;
        fin >> model_name;
        if (model_name != "LinearRegression") {
            cerr << "It is not a Linear Regression Model." << endl;
        }
        double temp;
        while (fin >> temp) {
            w.emplace_back(temp);
        }
    }
};

class LogitRegression {
   private:
    Vector w;

   public:
    double eta = 0.1;
    static double sigmoid(double x) { return 1.0 / (1 + exp(-x)); }
    static double sigmoid_diff(double x) {
        double temp = exp(-x);
        return temp / ((1 + temp) * (1 + temp));
    }
    void init(int n) {
        w.resize(n);
        for (auto &i : w) i = Rand() - 0.5;
    }
    LogitRegression() {}
    LogitRegression(int n) { init(n); }
    void show() const {
        cout << " y = sigmoid(";
        each_index(i, w) {
            cout << " "[i == 0] << "(" << w[i] << ") * x" << i << " "
                      << "+)"[i + 1 == (int)w.size()];
        }
        cout << endl;
    }
    double predict(const Vector &x) const {
        assert(x.size() == w.size());
        return sigmoid(Dot(x, w));
    }
    void train(const Vector &x, const double &y) {
        assert(x.size() == w.size());
        double s = Dot(x, w), y_ = sigmoid(s);
        w -= (eta * (y_ - y) * sigmoid_diff(s)) * x;
        for(auto i:w)if(isinf(i)){
            cerr<<"Divergence!"<<endl;
            exit(1);
        }
    }
    void save(const char *file_name) const {
        ofstream fout;
        fout.open(file_name, ios::out);
        fout << "LogitRegression" << endl;
        for (auto i : w) {
            fout << fixed << setprecision(10) << i << endl;
        }
    }
    void load(const char *file_name) {
        ifstream fin;
        fin.open(file_name, ios::in);
        w.clear();
        string model_name;
        fin >> model_name;
        if (model_name != "LogitRegression") {
            cerr << "It is not a Logit Regression Model." << endl;
        }
        double temp;
        while (fin >> temp) {
            w.emplace_back(temp);
        }
    }
};
#endif

#ifndef ML_Data_Reader
#define ML_Data_Reader

#include <bits/stdc++.h>
using namespace std;
#include "ML_Vector.h"

class DataSet {
   public:
    vector<Vector> data;
    void clear() { data.clear(); }
    void resize(int n, int m) {
        data.resize(n);
        for (auto &i : data) i = Vector(m, 0);
    }
    double &operator()(int x, int y) {
        if (x < 0 || x >= (int)data.size()) {
            cerr << "Row index out of range." << endl;
        } else if (y < 0 || y >= (int)data[x].size()) {
            cerr << "Col index out of range." << endl;
        }
        return data[x][y];
    }
    DataSet &operator+=(const DataSet &B) {
        if (B.data.size() != data.size()) {
            cerr << "Size error." << endl;
        }
        for (int i = 0; i < (int)data.size(); i++) {
            for (int j = 0; j < (int)B.data[i].size(); j++) {
                data[i].emplace_back(B.data[i][j]);
            }
        }
        return *this;
    }
    double mean(int c){
        double ans=0;
        int num=0;
        for(int i=0;i<(int)data.size();i++){
            if(!isnan(data[i][c])){
                ans+=data[i][c];
                num++;
            }
        }
        return ans/num;
    }
    double std_dev(int c){
        double ans=0,m=mean(c);
        int num=0;
        for(int i=0;i<(int)data.size();i++){
            if(!isnan(data[i][c])){
                ans+=(data[i][c]-m)*(data[i][c]-m);
                num++;
            }
        }
        return sqrt(ans/num);
    }
    double min(int c){
        double ans=data[0][c];
        for(int i=0;i<(int)data.size();i++)ans=std::min(ans,data[i][c]);
        return ans;
    }
    double max(int c){
        double ans=data[0][c];
        for(int i=0;i<(int)data.size();i++)ans=std::max(ans,data[i][c]);
        return ans;
    }
    void fill_nan_with_mean(){
        int R=data.size(),C=data[0].size();
        Vector m(C,0);
        for(int i=0;i<C;i++)m[i]=mean(i);
        for(int i=0;i<R;i++){
            for(int j=0;j<C;j++){
                if(isnan(data[i][j]))data[i][j]=m[j];
            }
        }
    }
    void min_max_normalization(int c){
        double mi=min(c),ma=max(c);
        for(int i=0;i<(int)data.size();i++)data[i][c]=(data[i][c]-mi)/(ma-mi);
    }
    void zscore_normalization(int c){
        double m=mean(c),s=std_dev(c);
        for(int i=0;i<(int)data.size();i++)data[i][c]=(data[i][c]-m)/s;
    }
    void show() const {
        static const int show_size = 5;
        cout << "row: " << data.size() << " col:" << data[0].size()
                  << endl;
        for (int i = 0; i < (int)data.size(); i++) {
            if (i >= show_size && i + show_size < (int)data.size()) {
                cout << "..." << endl;
                i = data.size() - show_size;
            }
            for (int j = 0; j < (int)data[i].size(); j++) {
                if (j >= show_size && j + show_size < (int)data[i].size()) {
                    cout << "\t...";
                    j = data[i].size() - show_size;
                }
                cout << "\t" << data[i][j];
            }
            cout << endl;
        }
        cout << endl;
    }
};

class CSV_Reader {
   private:
    static bool isNumber(const string &s) {
        int point = 0;
        int start=0;
        while(start<(int)s.size() && s[start]==' ')start++;
        if(start==(int)s.size())return false;
        if(s[start]=='-')start++;
        for (int i = start; i <= (int)s.size()-1; i++) {
            if(s[i]==' ')continue;
            if (!(s[i] >= '0' && s[i] <= '9') && s[i]!='.') {
                return false;
            } else if (s[i] == '.') {
                if (point == 1)
                    return false;
                else
                    point = 1;
            }
        }
        return true;
    }
    static double str2num(const string &s) {
        int point = 0, u = 0, start = 0;
        double v = 0, w = 1.0, sign = 1.0;
        while(start<(int)s.size() && s[start]==' ')start++;
        if(start==(int)s.size())return 0.0;
        if(s[start]=='-')start++,sign=-1.0;
        for (int i = start; i <= (int)s.size()-1; i++) {
            if(s[i]==' ')continue;
            if (!(s[i] >= '0' && s[i] <= '9') && s[i]!='.') {
                cerr << "Not number." << endl;
                exit(1);
            } else if (s[i] == '.') {
                if (point == 1) {
                    cerr << "2 or more points" << endl;
                    exit(1);
                } else {
                    point = 1;
                    v = u;
                }
            } else {
                if (point == 0)
                    u = u * 10 + s[i] - '0';
                else {
                    w *= 0.1;
                    v += w * (s[i] - '0');
                }
            }
        }
        return (point ? v : (u * 1.0)) * sign;
    }
    static vector<string> split(const string &s){
        vector<string> v;
        vector<int> p={-1};
        int flag=0;
        for(int i=0;i<(int)s.size();i++){
            if(s[i]=='\"')flag^=1;
            else if(s[i]==',' && flag==0)p.emplace_back(i);
        }
#ifdef WIN32
        p.emplace_back(s.size());
#else
        p.emplace_back((int)s.size()-1);
#endif
        for(int i=1;i<(int)p.size();i++){
            v.emplace_back(s.substr(p[i-1]+1,p[i]-p[i-1]-1));
        }
        return v;
    }
   public:
    ifstream file;
    vector<string> str_data;
    vector< vector<string> > data;
    vector<string> index;
    int file_flag = 0;
    vector<int> size()const{
        return (vector<int>){(int)data.size(),(int)data[0].size()};
    }
    void open(const char *file_name) {
        if (file_flag == 1) {
            file.close();
        }
        file_flag = 1;
        file.open(file_name, ios::in);
        if (!file) {
            cerr << "Open file error." << endl;
            return;
        }
        str_data.clear();
        static string temp;
        while (getline(file, temp)) {
            str_data.emplace_back(temp);
        }
        index = split(str_data[0]);
        int R = (int)str_data.size() - 1;
        data.resize(R);
        split(str_data[1]);
        for (int i = 1; i <= R; i++) {
            data[i-1]=split(str_data[i]);
        }
    }
    void describe() {
        int R = (int)str_data.size() - 1, C = data[0].size();
        cout << "row: " << R << " col: " << C << endl;
        vector<int> number(C, 0);
        unordered_map<string,int> num[C+1];
        for (int i = 0; i < R; i++) {
            for(int j=0;j<C;j++){
                if(isNumber(data[i][j]))number[j]++;
                num[j][data[i][j]]++;
            }
        }
        for (int i = 0; i < C; i++) {
            cout << "col " << i << ": " << index[i] << endl;
            cout << "     " << number[i] << " numbers  ";
            cout << num[i].size() << " categories  ";
            if(num[i][""]>0)cout << num[i][""] << " empty  ";
            else num[i].erase("");
            cout << endl;
            cout<<"     {";
            int j=1;
            for(auto it=num[i].begin();it!=num[i].end();it++){
                cout<<(it->first)<<", ";
                j++;
                if(j>4)break;
            }
            if(num[i].size()>4)cout<<"...";
            cout<<"}"<<endl;
        }
    }
    void shuffle() {
        random_shuffle(data.begin(), data.end());
    }
    void export_number_data(int r1, int r2, int c1, int c2, DataSet &D) {
        D.resize(r2 - r1 + 1, c2 - c1 + 1);
        for (int i = r1; i <= r2; i++) {
            for(int j=c1;j<=c2;j++){
                D(i - r1, j - c1)=str2num(data[i][j]);
                if(data[i][j].size()==0)D(i - r1, j - c1)=0.0/0.0;
            }
        }
    }
    void export_onehot_data(int r1, int r2, int c, DataSet &D) {
        static set<string> se;
        static map<string, int> mp;
        se.clear();
        mp.clear();
        for(int i=0;i<(int)data.size();i++)se.insert(data[i][c]);
        int tot=0;
        for (auto s : se) mp[s] = tot++;
        D.resize(r2 - r1 + 1, tot);
        for (int i = 0; i <= r2 - r1; i++) {
            D(i, mp[data[r1+i][c]]) = 1.0;
        }
    }
    void print_column(int c){
        for(int i=0;i<(int)data.size();i++){
            cout<<data[i][c]<<endl;
        }
    }
    void close() {
        if (file_flag == 0) {
            cerr << "No file is opened." << endl;
        }
        str_data.clear();
        data.clear();
        index.clear();
        file.close();
    }
};

class TXT_Reader {
   private:
    static bool isNumber(const string &s) {
        int point = 0;
        int start=0;
        while(start<(int)s.size() && s[start]==' ')start++;
        if(start==(int)s.size())return false;
        if(s[start]=='-')start++;
        for (int i = start; i <= (int)s.size()-1; i++) {
            if(s[i]==' ')continue;
            if (!(s[i] >= '0' && s[i] <= '9') && s[i]!='.') {
                return false;
            } else if (s[i] == '.') {
                if (point == 1)
                    return false;
                else
                    point = 1;
            }
        }
        return true;
    }
    static double str2num(const string &s) {
        int point = 0, u = 0, start = 0;
        double v = 0, w = 1.0, sign = 1.0;
        while(start<(int)s.size() && s[start]==' ')start++;
        if(start==(int)s.size())return 0.0;
        if(s[start]=='-')start++,sign=-1.0;
        for (int i = start; i <= (int)s.size()-1; i++) {
            if(s[i]==' ')continue;
            if (!(s[i] >= '0' && s[i] <= '9') && s[i]!='.') {
                cerr << "Not number." << endl;
                exit(1);
            } else if (s[i] == '.') {
                if (point == 1) {
                    cerr << "2 or more points" << endl;
                    exit(1);
                } else {
                    point = 1;
                    v = u;
                }
            } else {
                if (point == 0)
                    u = u * 10 + s[i] - '0';
                else {
                    w *= 0.1;
                    v += w * (s[i] - '0');
                }
            }
        }
        return (point ? v : (u * 1.0)) * sign;
    }
    static vector<string> split(const string &s){
        vector<string> v;
        vector<int> p={-1};
        int flag=0;
        for(int i=0;i<(int)s.size();i++){
            if(s[i]=='\"')flag^=1;
            else if(s[i]==',' && flag==0)p.emplace_back(i);
        }
#ifdef WIN32
        p.emplace_back(s.size());
#else
        p.emplace_back((int)s.size()-1);
#endif
        for(int i=1;i<(int)p.size();i++){
            v.emplace_back(s.substr(p[i-1]+1,p[i]-p[i-1]-1));
        }
        return v;
    }
   public:
    ifstream file;
    vector<string> str_data;
    vector< vector<string> > data;
    vector<string> index;
    int file_flag = 0;
    vector<int> size()const{
        return (vector<int>){(int)data.size(),(int)data[0].size()};
    }
    void open(const char *file_name) {
        if (file_flag == 1) {
            file.close();
        }
        file_flag = 1;
        file.open(file_name, ios::in);
        if (!file) {
            cerr << "Open file error." << endl;
            return;
        }
        str_data.clear();
        str_data.emplace_back("");
        static string temp;
        while (getline(file, temp)) {
            str_data.emplace_back(temp);
        }
        index = split(str_data[0]);
        int R = (int)str_data.size() - 1;
        data.resize(R);
        split(str_data[1]);
        for (int i = 1; i <= R; i++) {
            data[i-1]=split(str_data[i]);
        }
    }
    void describe() {
        int R = (int)str_data.size() - 1, C = data[0].size();
        cout << "row: " << R << " col: " << C << endl;
        vector<int> number(C, 0);
        unordered_map<string,int> num[C+1];
        for (int i = 0; i < R; i++) {
            for(int j=0;j<C;j++){
                if(isNumber(data[i][j]))number[j]++;
                num[j][data[i][j]]++;
            }
        }
        for (int i = 0; i < C; i++) {
            cout << "col " << i << ": " << index[i] << endl;
            cout << "     " << number[i] << " numbers  ";
            cout << num[i].size() << " categories  ";
            if(num[i][""]>0)cout << num[i][""] << " empty  ";
            else num[i].erase("");
            cout << endl;
            cout<<"     {";
            int j=1;
            for(auto it=num[i].begin();it!=num[i].end();it++){
                cout<<(it->first)<<", ";
                j++;
                if(j>4)break;
            }
            if(num[i].size()>4)cout<<"...";
            cout<<"}"<<endl;
        }
    }
    void shuffle() {
        random_shuffle(data.begin(), data.end());
    }
    void export_number_data(int r1, int r2, int c1, int c2, DataSet &D) {
        D.resize(r2 - r1 + 1, c2 - c1 + 1);
        for (int i = r1; i <= r2; i++) {
            for(int j=c1;j<=c2;j++){
                D(i - r1, j - c1)=str2num(data[i][j]);
                if(data[i][j].size()==0)D(i - r1, j - c1)=0.0/0.0;
            }
        }
    }
    void export_onehot_data(int r1, int r2, int c, DataSet &D) {
        static set<string> se;
        static map<string, int> mp;
        se.clear();
        mp.clear();
        for(int i=0;i<(int)data.size();i++)se.insert(data[i][c]);
        int tot=0;
        for (auto s : se) mp[s] = tot++;
        D.resize(r2 - r1 + 1, tot);
        for (int i = 0; i <= r2 - r1; i++) {
            D(i, mp[data[r1+i][c]]) = 1.0;
        }
    }
    void print_column(int c){
        for(int i=0;i<(int)data.size();i++){
            cout<<data[i][c]<<endl;
        }
    }
    void close() {
        if (file_flag == 0) {
            cerr << "No file is opened." << endl;
        }
        str_data.clear();
        data.clear();
        index.clear();
        file.close();
    }
};

#endif


#ifndef ML_Optimazer
#define ML_Optimazer

#include <bits/stdc++.h>
using namespace std;
//参数列表
class ParameterList{
public:
    vector<double*> w,dw;
    vector<bool*> frozen;
    void clear(){
        w.clear();
        dw.clear();
        frozen.clear();
    }
    void add_parameter(double &x,double &dx){
        w.emplace_back(&x);
        dw.emplace_back(&dx);
        frozen.emplace_back(new bool(false));
    }
    void add_parameter(double &x,double &dx,bool &frozen_flag){
        w.emplace_back(&x);
        dw.emplace_back(&dx);
        frozen.emplace_back(&frozen_flag);
    }
    void load(ifstream &f){
        for(auto &i:w)f>>(*i);
    }
    void save(ofstream &f){
        for(auto &i:w)f<<fixed<<setprecision(10)<<(*i)<<endl;
    }
};

//优化器
namespace Optimazer{
//梯度下降法
class GradientDescent{
public:
    double eta=0.05;
    GradientDescent(){}
    GradientDescent(double eta):eta(eta){}
    void init(const ParameterList &L){}
    void iterate(const ParameterList &L){
        int n=L.w.size();
        for(int i=0;i<n;i++)if(!(*L.frozen[i])){
            *(L.w[i])-=eta*(*(L.dw[i]));
        }
    }
};
class Adam{
public:
    double alpha=0.001,beta1=0.9,beta2=0.999,eps=1e-8;
    double power_beta1=1,power_beta2=1;
    vector<double> m,v,m_,v_;
    long long t=0;
    void init(const ParameterList &L){
        int n=L.w.size();
        m=v=m_=v_=vector<double>(n,0);
        t=0;
        power_beta1=1,power_beta2=1;
    }
    void load(ifstream &f){
        f>>t;
        for(auto &i:m)f>>i;
        for(auto &i:v)f>>i;
    }
    void save(ofstream &f){
        f<<t<<endl;
        for(auto &i:m)f<<fixed<<setprecision(10)<<i<<endl;
        for(auto &i:v)f<<fixed<<setprecision(10)<<i<<endl;
    }
    void iterate(const ParameterList &L){
        int n=L.w.size();
        t++;
#ifdef DEBUG
        for(int i=0;i<n;i++){
            assert(!isnan(*L.w[i]));
            assert(!isinf(*L.w[i]));
            assert(!isnan(*L.dw[i]));
            assert(!isinf(*L.dw[i]));
        }
#endif
        if(t<10000){
            power_beta1*=beta1,power_beta2*=beta2;
            for(int i=0;i<n;i++)if(!(*L.frozen[i])){
                m[i]=beta1*m[i]+(1.0-beta1)*(*L.dw[i]);
                v[i]=beta2*v[i]+(1.0-beta2)*(*L.dw[i])*(*L.dw[i]);
                m_[i]=m[i]/(1.0-power_beta1);
                v_[i]=v[i]/(1.0-power_beta2);
                (*L.w[i])-=alpha*m_[i]/(sqrt(v_[i])+eps);
            }
        }else{
            for(int i=0;i<n;i++)if(!(*L.frozen[i])){
                m[i]=beta1*m[i]+(1.0-beta1)*(*L.dw[i]);
                v[i]=beta2*v[i]+(1.0-beta2)*(*L.dw[i])*(*L.dw[i]);
                (*L.w[i])-=alpha*m[i]/(sqrt(v[i])+eps);
            }
        }
    }
};
template <const int batch>
class BatchAdam{
public:
    double alpha=0.001,beta1=0.9,beta2=0.999,eps=1e-8;
    double power_beta1=1,power_beta2=1;
    vector<double> m,v,m_,v_,sum_dw;
    long long t=0;
    void init(const ParameterList &L){
        int n=L.w.size();
        m=v=m_=v_=sum_dw=vector<double>(n,0);
        t=0;
        power_beta1=1,power_beta2=1;
    }
    void load(ifstream &f){
        f>>t;
        for(auto &i:m)f>>i;
        for(auto &i:v)f>>i;
    }
    void save(ofstream &f){
        f<<t<<endl;
        for(auto &i:m)f<<fixed<<setprecision(10)<<i<<endl;
        for(auto &i:v)f<<fixed<<setprecision(10)<<i<<endl;
    }
    void iterate(const ParameterList &L){
        static int cnt=0;
        int n=L.w.size();
        for(int i=0;i<n;i++)sum_dw[i]+=(*L.dw[i])*(1.0/batch);
        cnt++;
        if(cnt%batch!=0)return;
        t++;
#ifdef DEBUG
        for(int i=0;i<n;i++){
            assert(!isnan(*L.w[i]));
            assert(!isinf(*L.w[i]));
            assert(!isnan(*L.dw[i]));
            assert(!isinf(*L.dw[i]));
        }
#endif
        if(t<10000){
            power_beta1*=beta1,power_beta2*=beta2;
            for(int i=0;i<n;i++)if(!(*L.frozen[i])){
                m[i]=beta1*m[i]+(1.0-beta1)*(sum_dw[i]);
                v[i]=beta2*v[i]+(1.0-beta2)*(sum_dw[i])*(sum_dw[i]);
                m_[i]=m[i]/(1.0-power_beta1);
                v_[i]=v[i]/(1.0-power_beta2);
                (*L.w[i])-=alpha*m_[i]/(sqrt(v_[i])+eps);
            }
        }else{
            for(int i=0;i<n;i++)if(!(*L.frozen[i])){
                m[i]=beta1*m[i]+(1.0-beta1)*(sum_dw[i]);
                v[i]=beta2*v[i]+(1.0-beta2)*(sum_dw[i])*(sum_dw[i]);
                (*L.w[i])-=alpha*m[i]/(sqrt(v[i])+eps);
            }
        }
        sum_dw=vector<double>(n,0);
    }
};
}

#endif


const double init_L=-0.1,init_R=0.1;

//全连接边
template <const int N,const int M>
class ComplateEdge{
public:
    SmartArray<N,M> w,dw;
    ComplateEdge<N,M>(){w.reset_weight(init_L,init_R);}
    void reset_weight(double l,double r){w.reset_weight(l,r);}
    double& operator () (int x,int y){return w(x,y);}
    void get_parameters(ParameterList &PL){
        for(int i=0;i<N*M;i++)PL.add_parameter(w[i],dw[i]);
    }
};
template <class LayerType1,class LayerType2>
ComplateEdge<LayerType1::output_size,LayerType2::input_size> full_connect(LayerType1 &A,LayerType2 &B){
    ComplateEdge<LayerType1::output_size,LayerType2::input_size> E;
    return E;
}

/*损失函数(导数)*/
function<Vector(Vector,Vector)> mse=[](Vector y,Vector y_){
    each_index(i,y)y[i]=y_[i]-y[i];
    return y;
};
function<Vector(Vector,Vector)> mae=[](Vector y,Vector y_){
    each_index(i,y)y[i]=(y[i]<y_[i])?1.0:-1.0;
    return y;
};
function<Vector(Vector,Vector)> crossEntropy=[](Vector y,Vector y_){
    each_index(i,y)y[i]=-y[i]/y_[i]+(1-y[i])/(1-y_[i]);
    return y;
};
function<Vector(Vector,Vector)> singleCrossEntropy=[](Vector y,Vector y_){
    each_index(i,y)if(y[i]>0.5){
        y[i]=-1.0/y_[i];
        break;
    }
    return y;
};
function<Vector(Vector,Vector)> softmax_crossEntropy=[](Vector y,Vector y_){
    double sum=0;
    each_index(i,y)y_[i]=exp(y_[i]),sum+=y_[i];
    each_index(i,y){
        y_[i]=y_[i]/sum;
        if(y[i]>0.5)y_[i]-=1.0;
    }
    return y_;
};

//激活函数
function<double(double)> constant = [](double x) { return x; };
function<double(double)> constant_diff = [](double x) { return 1.0; };
function<double(double)> sigmoid = [](double x) {
    return 1.0 / (1.0 + exp(-x));
};
function<double(double)> sigmoid_diff = [](double x) {
    double temp = exp(-x);
    return temp / ((1.0 + temp) * (1.0 + temp));
};
function<double(double)> relu = [](double x) {
    return x>0.0?x:0.0;
};
function<double(double)> relu_diff = [](double x) {
    return x>0.0?1.0:0.0;
};
function<double(double)> LeakyReLU = [](double x) {
    return x>0.0?x:(0.01*x);
};
function<double(double)> LeakyReLU_diff = [](double x) {
    return x>0.0?1.0:0.01;
};
function<double(double)> Tanh = [](double x) {
    double a=exp(x),b=exp(-x);
    return (a-b)/(a+b);
};
function<double(double)> Tanh_diff = [](double x) {
    double a=exp(x)+exp(-x);
    return 2.0/(a*a);
};
//一元激活函数
class Activation{
public:
    function<double(double)> f,f_;
    Activation(){f=constant,f_=constant_diff;}
    Activation(function<double(double)> f,function<double(double)> f_):f(f),f_(f_){}
    virtual void calc(double x[],double y[],int n){
        for(int i=0;i<n;i++)y[i]=f(x[i]);
    }
    virtual void calc_diff(double x[],double y[],int n){
        for(int i=0;i<n;i++)y[i]=f_(x[i]);
    }
};

//层
class Layer{};

//Dense层
template <const int N>
class DenseLayer:public Layer{
public:
    Activation f;
    static const int input_size=N,output_size=N;
    SmartArray<1,N> in_val,out_val,in_diff,out_diff,c;
    int use_bias=0;
    void reset_weight(double l=init_L,double r=init_R){
        for(int i=0;i<N;i++)c[i]=Rand(l,r);
    }
    DenseLayer(){}
    DenseLayer(function<double(double)> f1,function<double(double)> f2){
        use_bias=1;
        reset_weight();
        f=Activation(f1,f2);
    }
    void clear(){
        in_val.clear();
        in_diff.clear();
    }
    void forward_solve(){
        if(use_bias==1){
            for(int i=0;i<N;i++)in_val[i]+=c[i];
        }
        f.calc(in_val.data,out_val.data,N);
    }
    void backward_solve(){
        static double temp[N];
        f.calc_diff(in_val.data,temp,N);
        for(int i=0;i<N;i++)out_diff[i]=in_diff[i]*temp[i];
    }
    void get_parameters(ParameterList &PL){
        if(use_bias==1){
            for(int i=0;i<N;i++)PL.add_parameter(c[i],out_diff[i]);
        }
    }
};

//Normalize层
template <const int N>
class NormalizeLayer:public Layer{
public:
    static const int input_size=N,output_size=N;
    SmartArray<1,N> in_val,out_val,in_diff,out_diff;
    double ma,mi;
    void clear(){
        in_val.clear();
        in_diff.clear();
    }
    void forward_solve(){
        ma=mi=in_val[0];
        for(int i=1;i<N;i++)mi=min(mi,in_val[i]),ma=max(ma,in_val[i]);
        double temp=1.0/(ma-mi);
        for(int i=0;i<N;i++)out_val[i]=(in_val[i]-mi)*temp;
    }
    void backward_solve(){
        double temp=1.0/(ma-mi);
        for(int i=0;i<N;i++)out_diff[i]=in_diff[i]*temp;
    }
    void get_parameters(ParameterList &PL){}
};

//卷积层
template <const int H_in,const int W_in,const int H_c,const int W_c>
class ConvLayer:public Layer{
public:
    static const int H_out=H_in-H_c+1,W_out=W_in-W_c+1;
    static const int input_size=H_in*W_in,output_size=H_out*W_out;
    SmartArray<H_c,W_c> w,dw;
    SmartArray<H_in,W_in> in_val,out_diff;
    SmartArray<H_out,W_out> out_val,in_diff;
    int use_bias=1;
    double c,dc;
    void reset_weight(double l=init_L,double r=init_R){
        w.reset_weight(l,r);
        c=Rand(l,r);
    }
    ConvLayer<H_in,W_in,H_c,W_c>(){
        reset_weight();
        use_bias=1;
    }
    void clear(){
        in_val.clear();
        in_diff.clear();
    }
    void forward_solve(){
        out_val.clear();
        for(int i=0;i<H_out;i++)for(int j=0;j<W_out;j++){
            for(int r=0;r<H_c;r++)for(int c=0;c<W_c;c++){
                out_val(i,j)+=w(r,c)*in_val(i+r,j+c);
            }
            out_val(i,j)+=c;
        }
    }
    void backward_solve(){
        for(int i=0;i<H_in;i++)for(int j=0;j<W_in;j++){
            out_diff[i*H_in+j]=0;
            for(int r=0;r<H_c;r++)for(int c=0;c<W_c;c++){
                int ii=i+r-H_c+1,jj=j+c-W_c+1;
                if(ii>=0 && ii<H_out && jj>=0 && jj<W_out){
                    out_diff[i*H_in+j]+=in_diff(ii,jj)*w(H_c-r-1,W_c-c-1);
                }
            }
        }
        dw.clear();
        dc=0;
        for(int i=0;i<H_out;i++)for(int j=0;j<W_out;j++){
            assert(!isnan(in_diff(i,j)));
            for(int r=0;r<H_c;r++)for(int c=0;c<W_c;c++){
                dw(r,c)+=in_diff(i,j)*in_val(i+r,j+c);
            }
            dc+=in_diff(i,j);
        }
    }
    void get_parameters(ParameterList &PL){
        for(int i=0;i<H_c;i++)for(int j=0;j<W_c;j++)PL.add_parameter(w(i,j),dw(i,j));
        if(use_bias)PL.add_parameter(c,dc);
    }
};

template <const int H_I,const int W_I,const int H_P,const int W_P,const int H_Z=0,const int W_Z=0>
class PaddingLayer:public Layer{
public:
#define H(x) H_##x
#define W(x) W_##x
    static const int H(O)=H(I)+2*H(P)+H(Z)*(H(I)-1),W(O)=W(I)+2*W(P)+W(Z)*(W(I)-1);
    static const int input_size=H(I)*W(I),output_size=H(O)*W(O);
    SmartArray<H(I),W(I)> in_val,out_diff;
    SmartArray<H(O),W(O)> out_val,in_diff;
    void clear(){
        in_val.clear();
        in_diff.clear();
    }
    void forward_solve(){
        out_val.clear();
        for(int i=0;i<H(I);i++)for(int j=0;j<W(I);j++){
            out_val(H(P)+i*(H(Z)+1),W(P)+j*(W(Z)+1))=in_val(i,j);
        }
    }
    void backward_solve(){
        for(int i=0;i<H(I);i++)for(int j=0;j<W(I);j++){
            out_diff(i,j)=in_diff(H(P)+i*(H(Z)+1),W(P)+j*(W(Z)+1));
        }
    }
    void get_parameters(ParameterList &PL){}
#undef H
#undef W
};

//二维卷积层
template <const int H_I,const int W_I,const int H_C,const int W_C,const int H_S=1,const int W_S=1>
class Conv2DLayer:public Layer{
public:
#define H(x) H_##x
#define W(x) W_##x
    static const int H(O)=(H(I)-H(C)+H(S))/H(S),W(O)=(W(I)-W(C)+W(S))/W(S);
    static const int input_size=H(I)*W(I),output_size=H(O)*W(O);
    SmartArray<H(C),W(C)> w,dw;
    SmartArray<H(I),W(I)> in_val,out_diff;
    SmartArray<H(O),W(O)> out_val,in_diff;
    int use_bias=1;
    double bias,d_bias;
    void reset_weight(double l=init_L,double r=init_R){
        w.reset_weight(l,r);
        bias=Rand(l,r);
    }
    Conv2DLayer<H_I,W_I,H_C,W_C,H_S,W_S>(){reset_weight();}
    void clear(){
        in_val.clear();
        in_diff.clear();
    }
    void forward_solve(){
        out_val.clear();
        for(int i=0;i<H(O);i++)for(int j=0;j<W(O);j++){
            for(int r=0;r<H(C);r++)for(int c=0;c<W(C);c++){
                out_val(i,j)+=w(r,c)*in_val(i*H(S)+r,j*W(S)+c);
            }
            out_val(i,j)+=bias;
        }
    }
    void backward_solve(){
        dw.clear();
        d_bias=0;
        for(int i=0;i<H(O);i++)for(int j=0;j<W(O);j++){
            for(int r=0;r<H(C);r++)for(int c=0;c<W(C);c++){
                dw(r,c)+=in_val(i*H(S)+r,j*W(S)+c)*in_diff(i,j);
                out_diff(i*H(S)+r,j*W(S)+c)+=w(r,c)*in_diff(i,j);
            }
            d_bias+=in_diff(i,j);
        }
    }
    void get_parameters(ParameterList &PL){
        for(int i=0;i<H(C);i++)for(int j=0;j<W(C);j++)PL.add_parameter(w(i,j),dw(i,j));
        if(use_bias)PL.add_parameter(bias,d_bias);
    }
#undef H
#undef W
};

/*最大值池化层*/
template <const int H_in,const int W_in,const int H_c,const int W_c>
class MaxPoolLayer:public Layer{
public:
    static const int H_out=H_in/H_c,W_out=W_in/W_c;
    static const int input_size=H_in*W_in,output_size=H_out*W_out;
    SmartArray<H_in,W_in> in_val,out_diff;
    SmartArray<H_out,W_out> out_val,in_diff;
    void clear(){
        in_val.clear();
        in_diff.clear();
    }
    void forward_solve(){
        for(int i=0;i<H_out;i++)for(int j=0;j<W_out;j++){
            out_val(i,j)=in_val(i*H_c,j*W_c);
            for(int r=0;r<H_c;r++)for(int c=0;c<W_c;c++){
                out_val(i,j)=max(out_val(i,j),in_val(i*H_c+r,j*W_c+c));
            }
        }
    }
    void backward_solve(){
        out_diff.clear();
        int max_i=0,max_j=0;
        for(int i=0;i<H_out;i++)for(int j=0;j<W_out;j++){
            for(int r=0;r<H_c;r++)for(int c=0;c<W_c;c++)if(out_val(i,j)==in_val(i*H_c+r,j*W_c+c)){
                max_i=i*H_c+r;
                max_j=j*W_c+c;
            }
            assert(max_i>=i*H_c && max_i<i*H_c+H_c && max_j>=j*W_c && max_j<j*W_c+W_c);
            out_diff(max_i,max_j)=in_diff(i,j);
        }
    }
    void get_parameters(ParameterList &PL){}
};

//平均池化层
template <const int H_in,const int W_in,const int H_c,const int W_c>
class AvePoolLayer:public Layer{
public:
    static const int H_out=H_in/H_c,W_out=W_in/W_c;
    static const int input_size=H_in*W_in,output_size=H_out*W_out;
    SmartArray<H_in,W_in> in_val,out_diff;
    SmartArray<H_out,W_out> out_val,in_diff;
    void clear(){
        in_val.clear();
        in_diff.clear();
    }
    void forward_solve(){
        static const double temp=1.0/(H_c*W_c);
        out_val.clear();
        for(int i=0;i<H_out;i++)for(int j=0;j<W_out;j++){
            for(int r=0;r<H_c;r++)for(int c=0;c<W_c;c++){
                out_val(i,j)+=in_val(i*H_c+r,j*W_c+c)*temp;
            }
        }
    }
    void backward_solve(){
        static const double temp=1.0/(H_c*W_c);
        out_diff.clear();
        for(int i=0;i<H_out;i++)for(int j=0;j<W_out;j++){
            for(int r=0;r<H_c;r++)for(int c=0;c<W_c;c++){
                out_diff(i*H_c+r,j*W_c+c)+=in_diff(i,j)*temp;
            }
        }
    }
    void get_parameters(ParameterList &PL){}
};

//并行层
template<class LayerType,const int N>
class Parallel:public Layer{
public:
    static const int input_size=LayerType::input_size*N;
    static const int output_size=LayerType::output_size*N;
    SmartArray<1,input_size> in_val,out_diff;
    SmartArray<1,output_size> in_diff,out_val;
    LayerType L[N];
    LayerType& operator [] (int x){return L[x];}
    void clear(){
        for(int i=0;i<N;i++)L[i].clear();
        in_val.clear();
        in_diff.clear();
    }
    void pass_in_val(){
        int tot=0;
        for(int i=0;i<N;i++){
            for(int j=0;j<LayerType::input_size;j++){
                L[i].in_val[j]=in_val[tot++];
            }
        }
    }
    void pass_out_val(){
        int tot=0;
        for(int i=0;i<N;i++){
            for(int j=0;j<LayerType::output_size;j++){
                out_val[tot++]=L[i].out_val[j];
            }
        }
    }
    void pass_in_diff(){
        int tot=0;
        for(int i=0;i<N;i++){
            for(int j=0;j<LayerType::output_size;j++){
                L[i].in_diff[j]=in_diff[tot++];
            }
        }
    }
    void pass_out_diff(){
        int tot=0;
        for(int i=0;i<N;i++){
            for(int j=0;j<LayerType::input_size;j++){
                out_diff[tot++]=L[i].out_diff[j];
            }
        }
    }
    void forward_solve(){
        pass_in_val();
        for(int i=0;i<N;i++)L[i].forward_solve();
        pass_out_val();
    }
    void backward_solve(){
        pass_in_diff();
        for(int i=0;i<N;i++)L[i].backward_solve();
        pass_out_diff();
    }
    void get_parameters(ParameterList &PL){
        for(int i=0;i<N;i++)L[i].get_parameters(PL);
    }
};
template<class LayerType,const int N>
class ExpandParallel:public Layer{
public:
    static const int input_size=LayerType::input_size;
    static const int output_size=LayerType::output_size*N;
    SmartArray<1,input_size> in_val,out_diff;
    SmartArray<1,output_size> in_diff,out_val;
    LayerType L[N];
    LayerType& operator [] (int x){return L[x];}
    void clear(){
        for(int i=0;i<N;i++)L[i].clear();
        in_val.clear();
        in_diff.clear();
    }
    void pass_in_val(){
        for(int i=0;i<N;i++){
            for(int j=0;j<LayerType::input_size;j++){
                L[i].in_val[j]=in_val[j];
            }
        }
    }
    void pass_out_val(){
        int tot=0;
        for(int i=0;i<N;i++){
            for(int j=0;j<LayerType::output_size;j++){
                out_val[tot++]=L[i].out_val[j];
            }
        }
    }
    void pass_in_diff(){
        int tot=0;
        for(int i=0;i<N;i++){
            for(int j=0;j<LayerType::output_size;j++){
                L[i].in_diff[j]=in_diff[tot++];
            }
        }
    }
    void pass_out_diff(){
        out_diff.clear();
        for(int i=0;i<N;i++){
            for(int j=0;j<LayerType::input_size;j++){
                out_diff[j]+=L[i].out_diff[j];
            }
        }
    }
    void forward_solve(){
        pass_in_val();
        for(int i=0;i<N;i++)L[i].forward_solve();
        pass_out_val();
    }
    void backward_solve(){
        pass_in_diff();
        for(int i=0;i<N;i++)L[i].backward_solve();
        pass_out_diff();
    }
    void get_parameters(ParameterList &PL){
        for(int i=0;i<N;i++)L[i].get_parameters(PL);
    }
};

//Dropout层
template<const int N>
class DropoutLayer:public Layer{
public:
    static const int input_size=N;
    static const int output_size=N;
    double p=0.5,k=2.0;
    SmartArray<1,N> in_val,out_diff;
    SmartArray<1,N> in_diff,out_val;
    bool drop[N];
    DropoutLayer<N>(){}
    DropoutLayer<N>(double p):p(p),k(1.0/(1.0-p)){}
    void set_drop_probability(double prob){
        p=prob;
        k=1.0/(1.0-p);
    }
    void clear(){
        in_val.clear();
        in_diff.clear();
        for(int i=0;i<N;i++)drop[i]=0;
    }
    void forward_solve(){
        for(int i=0;i<N;i++)if(Rand(0,1)<p)drop[i]=1;
        for(int i=0;i<N;i++)out_val[i]=drop[i]?0:in_val[i]*k;
    }
    void backward_solve(){
        for(int i=0;i<N;i++)out_diff[i]=drop[i]?0:in_diff[i]*k;
    }
    void get_parameters(ParameterList &PL){}
};

//乘法层
template <const int N>
class MultiplicationLayer:public Layer{
public:
    static const int input_size=N,output_size=N;
    SmartArray<1,N> A,B,out_val;
    SmartArray<1,N> in_diff,A_diff,B_diff;
    void clear(){
        A.clear();
        B.clear();
        in_diff.clear();
    }
    void forward_solve(){
        for(int i=0;i<N;i++)out_val[i]=A[i]*B[i];
    }
    void backward_solve(){
        for(int i=0;i<N;i++)A_diff[i]=in_diff[i]*B[i];
        for(int i=0;i<N;i++)B_diff[i]=in_diff[i]*A[i];
    }
    void get_parameters(ParameterList &PL){}
};

//正向传播
template<class LayerType1,class LayerType2>
void push_forward(LayerType1 &A,LayerType2 &B){
    assert(LayerType1::output_size==LayerType2::input_size);
    for(int i=0;i<LayerType1::output_size;i++)B.in_val[i]+=A.out_val[i];
}
//逆向传播
template<class LayerType1,class LayerType2>
void push_backward(LayerType1 &A,LayerType2 &B){
    assert(LayerType1::output_size==LayerType2::input_size);
    for(int i=0;i<LayerType1::output_size;i++)A.in_diff[i]+=B.out_diff[i];
}
//全连接
template<class LayerType1,class LayerType2>
void push_forward(LayerType1 &A,LayerType2 &B,ComplateEdge<LayerType1::output_size,LayerType2::input_size> &E){
    for(int i=0;i<LayerType1::output_size;i++){
        for(int j=0;j<LayerType2::input_size;j++){
            B.in_val[j]+=A.out_val[i]*E(i,j);
        }
    }
}
template<class LayerType1,class LayerType2>
void push_backward(LayerType1 &A,LayerType2 &B,ComplateEdge<LayerType1::output_size,LayerType2::input_size> &E){
    for(int j=0;j<LayerType2::input_size;j++){
        for(int i=0;i<LayerType1::output_size;i++){
            A.in_diff[i]+=B.out_diff[j]*E(i,j);
            E.dw(i,j)=B.out_diff[j]*A.out_val[i];
        }
    }
}
//乘法层传播
template <class LayerType1,class LayerType2,const int N>
void push_forward(LayerType1 &L1,LayerType2 &L2,MultiplicationLayer<N> &L3){
    assert(LayerType1::output_size==N && LayerType2::output_size==N);
    for(int i=0;i<N;i++)L3.A[i]+=L1.out_val[i];
    for(int i=0;i<N;i++)L3.B[i]+=L2.out_val[i];
}
template <class LayerType1,class LayerType2,const int N>
void push_backward(LayerType1 &L1,LayerType2 &L2,MultiplicationLayer<N> &L3){
    assert(LayerType1::output_size==N && LayerType2::output_size==N);
    for(int i=0;i<N;i++)L1.in_diff[i]+=L3.A_diff[i];
    for(int i=0;i<N;i++)L2.in_diff[i]+=L3.B_diff[i];
}

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
