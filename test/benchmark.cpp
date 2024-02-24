#include "emp-zk/emp-zk.h"
#include <iostream>
#include "emp-tool/emp-tool.h"
#include "linear_reg.h"
#include "log_reg.h"

using namespace emp;
using namespace std;

int port, party;
const int threads = 16;


void test_circuit_zk(BoolIO<NetIO> *ios[threads], int party) {
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    Integer a(32, 3, ALICE);
    Integer b(32, 2, ALICE);
    cout << (a - b).reveal<uint32_t>(PUBLIC) << endl;

    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
}

void test_inference(BoolIO<NetIO> *ios[threads], int party) {
    vector<float> x_1 = {1, 2, 3};
    vector<float> x_2 = {2, 4, 5};
    vector<float> w = {1, 2, 4};
    float b = 3;
    vector<vector<float>> x = {x_1, x_2};
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    inference(party, x, w, b);
    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
}

float rand_float(float min, float max, PRG *prg) {
    uint32_t tmp;
    prg->random_data(&tmp, sizeof(uint32_t));
    int in_range = tmp % int(max - min);
//    float scale = tmp / MAXFLOAT;
//    cout << "scale " << scale;
//    cout << min + scale * (max - min) << "\n";
//    cout << min + float(in_range) << "\n";
    return min + float(in_range);

}


//range in 0+- 0.05 * range
float get_small_noise(int range = 10, float increment = 0.001) {
    return 0;
    PRG prg;
    uint32_t tmp;
    prg.random_data(&tmp, sizeof(uint32_t));
//    cout << "noise: " << -increment * range + (((tmp % (2 * range)) + 1) * 0.05) << "\n";
//    cout << tmp << "\n ";
    return -increment * range + ((tmp % (2 * range)) * increment);

}

void print_data(vector<float> label, vector<vector<float>> samples) {
    cout << "\nx: ";
    for (int i = 0; i < label.size(); ++i) for (int j = 0; j < samples[0].size(); ++j) cout << samples[i][j] << " ";
    cout << "\ny: ";
    for (int i = 0; i < label.size(); ++i) cout << label[i] << " ";
    cout << "\n";

}

void test_new_learn(BoolIO<NetIO> *ios[threads], int party, int dim, int size) {
    int n_iter = 1000;
    int k = 10;
    n_iter = size / k;
    float learning_rate = 0.001;
    int divide_by = 1;
//    int size = 10;
//    int dim = 5;
    float max = 100;
    float min = -100;
    vector<float> labels;
    vector<vector<float>> samples;
    PRG prg;
    // prep dataset
    for (int i = 0; i < size; ++i) {
        vector<float> tmp;
        for (int j = 0; j < dim; ++j) {
            tmp.push_back(i + 1 + get_small_noise());
        }
        samples.push_back(tmp);
    }
    for (int i = 0; i < size; ++i) {
        labels.push_back((i + 1) + get_small_noise());
    }
//    print_data(labels, samples);
    auto t1 = clock_start();
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    learn(party, n_iter, k, samples, labels, learning_rate, divide_by);
    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
    if (party == ALICE) cout << size << ", " << dim << ", " << time_from(t1) / 1000.0 << " ms, ";
}


void
test_batch_log_learn(NetIO *io, BoolIO<NetIO> *ios[threads], int party, int m, int n, int k = 10, int steps = 1000) {
//    steps = n / k;
//    cout << "steps: " << steps << "\n";
    float learning_rate = 0.1;
    float sigma = 1.0;

    vector<float> labels;
    vector<vector<float>> samples;
    PRG prg;
    // prep dataset
    for (int i = 0; i < n; ++i) {
        vector<float> tmp;
        if (i < n / 2) {
            for (int j = 0; j < m; ++j) {
                tmp.push_back(1 + get_small_noise());
            }
            labels.push_back(0 + get_small_noise());

        } else {
            for (int j = 0; j < m; ++j) {
                tmp.push_back(10 + get_small_noise());
            }
            labels.push_back(1);
        }
        samples.push_back(tmp);
    }
//    for (int i = 0; i < n; ++i) {
//        labels.push_back((i + 1) + get_small_noise());
//    }
//    print_data(labels, samples);
    auto t1 = clock_start();
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    log_reg_learn(io, party, steps, samples, labels, learning_rate, sigma, k);
    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
    if (party == ALICE) cout << n << ", " << m << ", " << time_from(t1) / 1000.0  << " ms, ";
}

void test_batch_ftrl(NetIO *io, BoolIO<NetIO> *ios[threads], int party, int m, int n, int k = 10, int steps = 1000) {
//    steps = n / k;
//    cout << "steps: " << steps << "\n";
    float learning_rate = 0.1;
    float sigma = 1.0;

    vector<float> labels;
    vector<vector<float>> samples;
    PRG prg;
// prep dataset
    for (int i = 0; i < n; ++i) {
        vector<float> tmp;
        if (i < n / 2) {
            for (int j = 0; j < m; ++j) {
                tmp.push_back(1 + get_small_noise());
            }
            labels.push_back(0 + get_small_noise());

        } else {
            for (int j = 0; j < m; ++j) {
                tmp.push_back(10 + get_small_noise());
            }
            labels.push_back(1);
        }
        samples.push_back(tmp);
    }
    auto t1 = clock_start();
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    dp_ftrl(io, party, steps, samples, labels, learning_rate, sigma, k);
    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
    if (party == ALICE) cout << n << ", " << m << ", " << time_from(t1) / 1000.0  << " ms, ";
}


void test_learn(BoolIO<NetIO> *ios[threads], int party) {
    int n_iter = 1000;
    int k = 4;
    float learning_rate = 0.0001;
    int divide_by = 1;
    vector<float> labels = {1, 2, 3, 4, 5, 6, 7, 8};
    vector<float> f1 = {1};
    vector<float> f2 = {2.1};
    vector<float> f3 = {3.1};
    vector<float> f4 = {3.91};
    vector<float> f5 = {4.95};
    vector<float> f6 = {5.95};
    vector<float> f7 = {6.95};
    vector<float> f8 = {8.02};
    vector<vector<float>> samples = {f1, f2, f3, f4, f5, f6, f7, f8};
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    learn(party, n_iter, k, samples, labels, learning_rate, divide_by);
    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
}

void test_log_learn(NetIO *io, BoolIO<NetIO> *ios[threads], int party) {
    int steps = 1000;
    float learning_rate = 0.1;
    float sigma = 1.0;
    int k = 10;
    vector<float> labels = {0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1};
    vector<float> sample_list = {0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50,
                                 4.00,
                                 4.25, 4.50, 4.75, 5.00, 5.50};
    vector<vector<float>> samples;
    for (int i = 0; i < sample_list.size(); ++i) {
        vector<float> tmp = {sample_list[i], 1};
        samples.push_back(tmp);
    }
    setup_zk_bool<BoolIO<NetIO>>(ios, threads, party);
    log_reg_learn(io, party, steps, samples, labels, learning_rate, sigma, k);
    bool cheat = finalize_zk_bool<BoolIO<NetIO>>();
    if (cheat)error("cheat!\n");
}

void bench(NetIO *io, BoolIO<NetIO> *ios[threads], int party, int dim, int size, uint64_t &previous_data,
           uint64_t &total_data,
           int k = 10, int steps = 1000) {
//    test_new_learn(ios, party, dim, size);
    test_batch_log_learn(io, ios, party, dim, size, k, steps);
    previous_data = total_data;
    total_data = 0;
    for (int i = 0; i < threads; ++i) {
        total_data += ios[i]->counter;
    }
    if (party == ALICE)cout << (total_data - previous_data) / 1000.0  << " KB\n";
}

void bench_ftrl(NetIO *io, BoolIO<NetIO> *ios[threads], int party, int dim, int size, uint64_t &previous_data,
                uint64_t &total_data,
                int k = 10, int steps = 1000) {
//    test_new_learn(ios, party, dim, size);
    test_batch_ftrl(io, ios, party, dim, size, k, steps);
    previous_data = total_data;
    total_data = 0;
    for (int i = 0; i < threads; ++i) {
        total_data += ios[i]->counter;
    }
    if (party == ALICE)cout << (total_data - previous_data) / 1000.0  << " KB\n";
}

int main(int argc, char **argv) {
    parse_party_and_port(argv, &party, &port);
    BoolIO<NetIO> *ios[threads];
    for (int i = 0; i < threads; ++i)
        ios[i] = new BoolIO<NetIO>(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
    NetIO *randomness_io = new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + 1000);
//    std::cout << "Current path is " << fs::current_path() << '\n'; // (1)
//    std::cout << "Current path is " << fs::current_path() << '\n';
//    test_learn(ios, party);
//    test_inference(ios, party);
    uint64_t total_data = 0;
    uint64_t previous_data = 0;
//    if (party == ALICE)cout << "size, dim, time, comm\n";
//    test_learn(ios, party);
//    test_log_learn(ios, party);

    bench(randomness_io, ios, party, 1 << 4, 100, previous_data, total_data, 5, 20);
//    bench_ftrl(randomness_io, ios, party, 1 << 4, 100, previous_data, total_data, 1, 1);
//    bench_ftrl(randomness_io, ios, party, 1 << 6, 1, previous_data, total_data, 1, 100);
//    bench_ftrl(randomness_io, ios, party, 1 << 7, 1, previous_data, total_data, 1, 100);
//    bench_ftrl(randomness_io, ios, party, 1 << 8, 1, previous_data, total_data, 1, 100);
//    bench_ftrl(randomness_io, ios, party, 1 << 9, 1, previous_data, total_data, 1, 100);
//    bench_ftrl(randomness_io, ios, party, 1 << 10, 1, previous_data, total_data, 1, 100);
//    bench_ftrl(randomness_io, ios, party, 1 << 11, 1, previous_data, total_data, 1, 100);
//    bench_ftrl(randomness_io, ios, party, 1 << 12, 1, previous_data, total_data, 1, 100);
//    bench_ftrl(randomness_io, ios, party, 1 << 13, 1, previous_data, total_data, 1, 100);
//randomness_io

//    bench(randomness_io, ios, party, 4096, 20, previous_data, total_data, 1, 20);  //SimCLRv2
//    bench(randomness_io, ios, party, 1024, 20, previous_data, total_data, 1, 20);  //ResNeXt
//    bench(randomness_io, ios, party, 3969, 20, previous_data, total_data, 1, 20);  //ScatterNet
//    bench(randomness_io, ios, party, 15552, 5, previous_data, total_data, 1, 5);  //ScatterNet
//    bench(ios, party, 3969, 1024, previous_data, total_data, 1024, 1);  //FASHION-MNIST
//    bench(randomness_io, ios, party, 15552, 256, previous_data, total_data, 256, 1);  //CIFAR-10
//    get_common_randomness(randomness_io, party);




    for (int i = 0; i < threads; ++i) {
        delete ios[i]->io;
        delete ios[i];
    }
    return 0;
}
