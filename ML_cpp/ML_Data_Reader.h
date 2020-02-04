#include <bits/stdc++.h>
#ifndef ML_Data_Reader
#define ML_Data_Reader

class DataSet {
   public:
    std::vector<Vector> data;
    void clear() { data.clear(); }
    void resize(int n, int m) {
        data.resize(n);
        for (auto &i : data) i = Vector(m, 0);
    }
    double &operator()(int x, int y) {
        if (x < 0 || x >= (int)data.size()) {
            std::cerr << "Row index out of range." << std::endl;
        } else if (y < 0 || y >= (int)data[x].size()) {
            std::cerr << "Col index out of range." << std::endl;
        }
        return data[x][y];
    }
    DataSet &operator+=(const DataSet &B) {
        if (B.data.size() != data.size()) {
            std::cerr << "Size error." << std::endl;
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
            if(!std::isnan(data[i][c])){
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
            if(!std::isnan(data[i][c])){
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
                if(std::isnan(data[i][j]))data[i][j]=m[j];
            }
        }
    }
    void min_max_normalization(int c){
        double mi=min(c),ma=max(c);
        for(int i=0;i<(int)data.size();i++)data[i][c]=(data[i][c]-mi)/(ma-mi);
    }
    void show() const {
        static const int show_size = 5;
        std::cout << "row: " << data.size() << " col:" << data[0].size()
                  << std::endl;
        for (int i = 0; i < (int)data.size(); i++) {
            if (i >= show_size && i + show_size < (int)data.size()) {
                std::cout << "..." << std::endl;
                i = data.size() - show_size;
            }
            for (int j = 0; j < (int)data[i].size(); j++) {
                if (j >= show_size && j + show_size < (int)data[i].size()) {
                    std::cout << "\t...";
                    j = data[i].size() - show_size;
                }
                std::cout << "\t" << data[i][j];
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
};

class CSV_Reader {
   private:
    std::ifstream file;
    std::vector<std::string> str_data;
    std::vector< std::vector<std::string> > data;
    std::vector<std::string> index;
    int file_flag = 0;

   public:
    bool isNumber(const std::string &s) {
        int point = 0;
        for (int i = 0; i <= (int)s.size()-1; i++) {
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
    double str2num(const std::string &s) {
        int point = 0, u = 0;
        double v = 0, w = 1.0;
        for (int i = 0; i <= (int)s.size()-1; i++) {
            if (!(s[i] >= '0' && s[i] <= '9') && s[i]!='.') {
                std::cerr << "Not number." << std::endl;
                return -1.0;
            } else if (s[i] == '.') {
                if (point == 1) {
                    std::cerr << "2 or more points" << std::endl;
                    return -1.0;
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
        return point ? v : (u * 1.0);
    }
    void open(const char *file_name) {
        if (file_flag == 1) {
            file.close();
        }
        file_flag = 1;
        file.open(file_name, std::ios::in);
        if (!file) {
            std::cerr << "Open file error." << std::endl;
            return;
        }
        str_data.clear();
        static std::string temp;
        while (getline(file, temp)) {
            str_data.emplace_back(temp);
        }
        index = (std::vector<std::string>){""};
        for (auto c : str_data[0]) {
            if (c == ',')
                index.emplace_back("");
            else
                index.back() += c;
        }
        //split
        int R = (int)str_data.size() - 1;
        data.resize(R);
        for (int i = 1; i <= R; i++) {
            int last = 0, idx = 0, flag=0;
            for (int j = 0; j <= (int)str_data[i].size(); j++) {
                if(flag==1){
                    if(str_data[i][j]=='\"')flag=0;
                }else if(str_data[i][j]=='\"'){
                    flag=1;
                }else if (j == (int)str_data[i].size() || str_data[i][j] == ',') {
                    data[i-1].push_back(str_data[i].substr(last,j-last));
                    last = j + 1;
                    idx++;
                }
            }
        }
    }
    void describe() {
        int R = (int)str_data.size() - 1, C = data[0].size();
        std::cout << "row: " << R << " col: " << C << std::endl;
        std::vector<int> number(C, 0);
        std::unordered_map<std::string,int> num[C+1];
        for (int i = 0; i < R; i++) {
            for(int j=0;j<C;j++){
                if(isNumber(data[i][j]))number[j]++;
                num[j][data[i][j]]++;
            }
        }
        for (int i = 0; i < C; i++) {
            std::cout << "col " << i << ": " << index[i] << std::endl;
            std::cout << "     " << number[i] << " numbers  ";
            std::cout << num[i].size() << " categories  ";
            if(num[i][""]>0)std::cout << num[i][""] << " empty  ";
            else num[i].erase("");
            std::cout << std::endl;
            std::cout<<"     {";
            int j=1;
            for(auto it=num[i].begin();it!=num[i].end();it++){
                std::cout<<(it->first)<<", ";
                j++;
                if(j>4)break;
            }
            if(num[i].size()>4)std::cout<<"...";
            std::cout<<"}"<<std::endl;
        }
    }
    void shuffle() {
        std::random_shuffle(str_data.begin() + 1, str_data.end());
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
        static std::set<std::string> se;
        static std::map<std::string, int> mp;
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
            std::cout<<data[i][c]<<std::endl;
        }
    }
    void close() {
        if (file_flag == 0) {
            std::cerr << "No file is opened." << std::endl;
        }
        file.close();
    }
};

#endif
