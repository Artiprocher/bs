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
        random_shuffle(str_data.begin() + 1, str_data.end());
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
