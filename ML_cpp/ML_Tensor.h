#include <bits/stdc++.h>
#ifndef ML_Tensor
#define ML_Tensor
class Tensor {
public:
    std::vector<double> value;
    std::vector<int> size;
    std::vector<int> num;
    void resize(std::vector<int> new_size) {
        size = new_size;
        num = new_size;
        for (int i = (int)num.size() - 1; i >= 0; i--) {
            if (i + 1 == (int)num.size()) {
                num[i] = 1;
            } else {
                num[i] = num[i + 1] * size[i + 1];
            }
        }
        int all = 1;
        for (auto &i : size)
            all *= i;
        value.resize(all);
    }
    double &operator () (int start_index, ...) {
        assert(start_index < size[0] && start_index >= 0);
        int position = start_index * num[0];
        va_list ls;
        va_start(ls, start_index);
        for (int i = 1; i < size.size(); i++) {
            int now = va_arg(ls, int);
            assert(now < size[i] && now >= 0);
            position += num[i] * now;
        }
        return value[position];
    }
};
#endif
