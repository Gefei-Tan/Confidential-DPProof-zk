#ifndef ZK_LR_LINEAR_REG_H
#define ZK_LR_LINEAR_REG_H

#include "emp-zk/emp-zk.h"
#include <iostream>
#include "emp-tool/emp-tool.h"

void permute_samples(vector<vector<float>> &samples, vector<float> &labels, vector<Bit> seed) {
    emp::AES_128_CTR_Calculator aes_128_ctr_calculator = emp::AES_128_CTR_Calculator();
    int size = samples.size();
    for (int k = 0; k < size; k++) {
        int idx = k % 4;
        vector<Bit> truncated(seed.begin() + 32 * idx, seed.begin() + 32 * (idx + 1));
        Integer tmp(32, 0);
        tmp.bits = truncated;
        int r = k + tmp.reveal<uint32_t>(PUBLIC) % (size - k);
        vector<float> tmp_sample = samples[k];
        samples[k] = samples[r];
        samples[r] = tmp_sample;

        float tmp_lbl = labels[k];
        labels[k] = labels[r];
        labels[r] = labels[k];
        swap(samples[k], samples[r]);
        swap(labels[k], labels[r]);
        if (idx == 3) {
            Integer ctr(128, k, PUBLIC);
            vector<Bit> ctr_bits = ctr.bits;
            aes_128_ctr_calculator.aes_128_ctr(&(seed[0].bit), &(seed[0].bit), &(ctr_bits[0].bit),
                                               nullptr, 128, PUBLIC, 0);
        }
    }
}


vector<Integer> flatten(vector<vector<Float>> &v) {
    int size = v[0].size();
    vector<Integer> res;
    for (auto i: v) {
        vector<Bit> flatten_bits;
        Integer tmp(size * 32, 0, PUBLIC);
        for (auto j: i) {
            for (auto k: j.value) {
                flatten_bits.push_back(k);
            }
        }
        tmp.bits = flatten_bits;
        res.push_back(tmp);
    }
    return res;
}

vector<vector<Float>> unflatten(vector<Integer> &v) {
    vector<vector<Float>> res;
    for (auto i: v) {
        vector<Float> tmp;
        for (int j = 0; j < i.bits.size(); j += 32) {
            vector<Bit> bits(i.bits.begin() + j, i.bits.begin() + j + 32);
            Float tmp_f(0, PUBLIC);
            for (int k = 0; k < 32; ++k) {
                tmp_f.value[k] = bits[k];
            }
            tmp.push_back(tmp_f);
        }
        res.push_back(tmp);
    }
    return res;

}

void permute_samples(vector<vector<Float>> &samples, vector<Float> &labels, vector<Bit> seed) {
    emp::AES_128_CTR_Calculator aes_128_ctr_calculator = emp::AES_128_CTR_Calculator();
    int size = samples.size();
    vector<Integer> random_key;
    // generate random key for permutation
    for (int k = 0; k < size; k++) {
        int idx = k % 4;
        vector<Bit> truncated(seed.begin() + 32 * idx, seed.begin() + 32 * (idx + 1));
        Integer tmp(32, 0);
        tmp.bits = truncated;
        random_key.push_back(tmp);
        if (idx == 3) {
            Integer ctr(128, k, PUBLIC);
            vector<Bit> ctr_bits = ctr.bits;
            aes_128_ctr_calculator.aes_128_ctr(&(seed[0].bit), &(seed[0].bit), &(ctr_bits[0].bit),
                                               nullptr, 128, PUBLIC, 0);
        }
    }

    vector<Integer> flatten_samples = flatten(samples);
    sort(random_key.data(), size, labels.data());
    sort(random_key.data(), size, flatten_samples.data());
    samples = unflatten(flatten_samples);

}


void inference(int party, vector<vector<float>> x, vector<float> w, float b) {
    vector<vector<Float>> _x;
    vector<Float> _w;
    // convert x,w,b to Float
    for (int i = 0; i < x.size(); ++i) {
        vector<Float> tmp;
        for (int j = 0; j < x[0].size(); ++j) tmp.push_back(Float(x[i][j], ALICE));
        _x.push_back(tmp);
    }
    for (int i = 0; i < w.size(); ++i) _w.push_back(Float(w[i], ALICE));
    Float _b = Float(b, ALICE);

    vector<Float> y;

    for (int i = 0; i < _x.size(); ++i) y.push_back(inner_product(begin(_w), end(_w), begin(_x[i]), _b));
    for (int i = 0; i < y.size(); ++i) cout << y[i].reveal<double>(PUBLIC) << " ";
}

double public_print(Float f) {
    return f.reveal<double>(PUBLIC);
}

void learn(int party, int n_iter, int k, vector<vector<float>> samples,
           vector<float> labels, float learning_rate, int divide_by) {
    int sample_len = samples[0].size();
//    cout << "sample len: " << sample_len << "\n";
    vector<Float> w(sample_len, Float(0, PUBLIC));
    Float b = Float(0, PUBLIC);
//    permute_samples(samples, labels);
    Float _learning_rate(learning_rate, PUBLIC);
    Float _k(k, PUBLIC);
    Float _divide_by(divide_by, PUBLIC);
    for (int cur_iter = 0; cur_iter < n_iter; ++cur_iter) {
//        permute_samples(samples, labels);
        vector<vector<float>> x_tmp(samples.begin(), samples.begin() + k); // randomly sample k
        vector<float> y_tmp(labels.begin(), labels.begin() + k);
        vector<vector<Float>> x;
        vector<Float> y;
        // convert to Float
        for (int i = 0; i < k; ++i) {
            vector<Float> tmp;
            for (int j = 0; j < x_tmp[0].size(); ++j) tmp.push_back(Float(x_tmp[i][j], ALICE));
            x.push_back(tmp);
            y.push_back(Float(y_tmp[i], ALICE));
        }
        // initialize w_grad and b_grad
        vector<Float> w_grad(sample_len, Float(0, PUBLIC));
        Float b_grad = Float(0, PUBLIC);

        for (int i = 0; i < k; ++i) {
            Float prediction = inner_product(begin(x[i]), end(x[i]), begin(w), b);
            Float tmp_w_grad_calc = Float(-2, PUBLIC) * (y[i] - prediction);
            for (int z = 0; z < sample_len; ++z) w_grad[z] = w_grad[z] + (tmp_w_grad_calc * x[i][z]);
            b_grad = b_grad + Float(-2, PUBLIC) * (y[i] - prediction);
        }

        for (int z = 0; z < w.size(); ++z) w[z] = w[z] - (_learning_rate * (w_grad[z] / _k));
        b = b - _learning_rate * (b_grad / _k);
        _learning_rate = _learning_rate / _divide_by;
    }
    cout << "final w: ";
    for (int i = 0; i < sample_len; ++i) cout << w[i].reveal<double>(PUBLIC) << ", ";
    cout << "\n";
    cout << "\n final b: " << b.reveal<double>(PUBLIC) << "\n";
}


#endif //ZK_LR_LINEAR_REG_H
