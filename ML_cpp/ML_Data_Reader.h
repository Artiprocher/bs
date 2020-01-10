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
    std::vector<std::string> index;
    int file_flag = 0;

   public:
    bool isNumber(const std::string &s, int l, int r) {
        if (r < l) return false;
        int point = 0;
        for (int i = l; i <= r; i++) {
            if (s[i] < '0' || s[i] > '9') {
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
    double str2num(const std::string &s, int l, int r) {
        int point = 0, u = 0;
        double v = 0, w = 1.0;
        for (int i = l; i <= r; i++) {
            if (s[i] < '0' || s[i] > '9') {
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
    }
    void describe() {
        int R = (int)str_data.size() - 1, C = 1;
        for (auto i : str_data[0]) {
            if (i == ',') C++;
        }
        std::cout << "row: " << R << " col: " << C << std::endl;
        std::vector<int> number(C, 0);
        for (int i = 1; i <= R; i++) {
            int last = 0, idx = 0;
            for (int j = 0; j <= (int)str_data[i].size(); j++) {
                if (j == (int)str_data[i].size() || str_data[i][j] == ',') {
                    number[idx] += isNumber(str_data[i], last, j - 1);
                    last = j + 1;
                    idx++;
                }
            }
        }
        for (int i = 0; i < C; i++) {
            std::cout << "col " << i << ": " << index[i] << " " << number[i]
                      << " numbers." << std::endl;
        }
    }
    void shuffle() {
        std::random_shuffle(str_data.begin() + 1, str_data.end());
    }
    void export_number_data(int r1, int r2, int c1, int c2, DataSet &D) {
        D.resize(r2 - r1 + 1, c2 - c1 + 1);
        for (int i = r1; i <= r2; i++) {
            int last = 0, idx = 0;
            for (int j = 0; j <= (int)str_data[i].size(); j++) {
                if (j == (int)str_data[i].size() || str_data[i][j] == ',') {
                    if (idx >= c1 && idx <= c2) {
                        D(i - r1, idx - c1) = str2num(str_data[i], last, j - 1);
                    }
                    last = j + 1;
                    idx++;
                }
            }
        }
    }
    void export_onehot_data(int r1, int r2, int c, DataSet &D) {
        static std::vector<std::string> a;
        a.resize(r2 - r1 + 1);
        for (int i = r1; i <= r2; i++) {
            int last = 0, idx = 0;
            for (int j = 0; j <= (int)str_data[i].size(); j++) {
                if (j == (int)str_data[i].size() || str_data[i][j] == ',') {
                    if (idx == c) {
                        a[i - r1] = str_data[i].substr(last, j - last);
                        break;
                    }
                    last = j + 1;
                    idx++;
                }
            }
        }
        static std::set<std::string> se;
        static std::map<std::string, int> mp;
        se.clear();
        mp.clear();
        for (auto s : a) se.insert(s);
        int tot = 0;
        for (auto s : se) mp[s] = tot++;
        D.resize(r2 - r1 + 1, tot);
        for (int i = 0; i < r2 - r1; i++) {
            D(i, mp[a[i]]) = 1.0;
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
