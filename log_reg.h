#ifndef ZK_LR_LOG_REG_H
#define ZK_LR_LOG_REG_H

#include "emp-zk/emp-zk.h"
#include <iostream>
#include "emp-tool/emp-tool.h"

static float C = 1;


pair<Float, Float> alt_gen_gaussian_noise(vector<Bit> seed, float mu, float sigma, bool first_half = true) {

    Float _u_1(0.1, PUBLIC);
    Float _u_2(0.2, PUBLIC);
    if (first_half) {
        for (int i = 0; i < 32; ++i) {
            _u_1.value[i] = seed[i];
            _u_2.value[i] = seed[i + 32];
        }
    } else {
        for (int i = 0; i < 32; ++i) {
            _u_1.value[i] = seed[i + 64];
            _u_2.value[i] = seed[i + 96];
        }
    }

    Float s = _u_1.sqr() * _u_2.sqr();

    Float NEG_2 = Float(-2, PUBLIC);
    Float _sigma = Float(sigma, PUBLIC);
    Float _mu = Float(mu, PUBLIC);
    Float Z_0 = _sigma * _u_1 * ((NEG_2 * s.ln()) / s).sqrt() + mu;
    Float Z_1 = _sigma * _u_2 * ((NEG_2 * s.ln()) / s).sqrt() + mu;
    return std::make_pair(Z_0, Z_1);
}

Float _sigmoid(Float z) {
    Float one = Float(1, PUBLIC);
    Float ret = one / (one + z.exp());;
    return ret;
}

Float get_l2_norm(vector<Float> vec) {
    int m = vec.size();
    Float sum(0, PUBLIC);
    for (int i = 0; i < m; ++i) {
        sum = sum + vec[i] ^ 2;
    }
    Float l2_norm = sum.sqrt();
    return l2_norm;
}

vector<Float> projection(vector<Float> theta, float radius = 1) {
    Float _radius = Float(radius, PUBLIC);
    // compute L2 norm
    int m = theta.size();
    Float sum(0, PUBLIC);
    for (int i = 0; i < m; ++i) {
        sum = sum + theta[i] ^ 2;
    }
    Float l2_norm = sum.sqrt();
    Bit flag = l2_norm.less_equal(_radius);
    if (!flag.reveal(PUBLIC)) {
        for (int i = 0; i < m; ++i) {
            theta[i] = theta[i] * radius / l2_norm;
        }
    }
    return theta;

}


vector<Float> _grad(vector<vector<Float>> x, vector<Float> y, vector<Float> theta) {
//    Float _C = Float(C, PUBLIC);
    Float zero = Float(0, PUBLIC);
    int n = x.size();
    int m = theta.size();
    vector<Float> final_results(m, zero);
    for (int i = 0; i < n; ++i) {
        vector<Float> results(m, zero);
        // compute per example gradient
        Float output = _sigmoid(inner_product(begin(x[i]), end(x[i]), begin(theta), zero));
        Float value = output - y[i];
        // update each gradient according to the i-th sample  
        for (int j = 0; j < m; ++j) {
            results[j] = results[j] + (x[i][j] * value);
        }
        Float l2_norm = get_l2_norm(results);
        //clipping per example
        for (int j = 0; j < m; ++j) {
            Bit flag = l2_norm.less_equal(results[j]);
            if (!flag.reveal()) {
                results[j] = results[j] / l2_norm;
            }
        }
        // sum all gradient
        for (int j = 0; j < m; ++j) {
            final_results[j] = final_results[j] + results[j];
        }
    }
    final_results = projection(final_results, 1);
    return final_results;
}


block get_common_randomness(NetIO *io, int party) {
    if (party == ALICE) {
        PRG prg;
        block r;
        prg.random_block(&r, 1);
        block commit_r[2];
        Hash::hash_once(&commit_r, &r, sizeof(r));
        io->send_block(commit_r, 2);
        block s;
        block verifier_digest[2];
        io->recv_block(&s, 1);
        io->recv_block(verifier_digest, 2);
        return r ^ s;

    } else {
        block recv_commit_r[2];
        block s;
        PRG prg;
        prg.random_block(&s, 1);
        io->recv_block(recv_commit_r, 2);
        block verifier_digest[2];
        Hash hash;
        hash.put(&s, sizeof(block));
        hash.put(&recv_commit_r, 2 * sizeof(block));
        hash.digest(&verifier_digest);
        io->send_block(&s, 1);
        io->send_block(verifier_digest, 2);
        io->flush();
        return zero_block;
    }
}


void log_reg_learn(NetIO *io, int party, int steps, vector<vector<float>> samples,
                   vector<float> labels, float learning_rate, float sigma, int k, float bounded_set_radius = 1) {
    int n = samples.size();    // "n" sample size
    int m = samples[0].size(); // "m" sample dimension
    // preparing seed
    block randomness_seed = get_common_randomness(io, party);
    bool seed_in_bits[128];
    block_to_bool(seed_in_bits, randomness_seed);
    vector<Bit> final_seed;
    final_seed.reserve(128);
    for (int i = 0; i < 128; ++i) {
        final_seed.push_back(Bit(seed_in_bits[i], ALICE));
    }
    vector<Float> theta(m, Float(0, PUBLIC));
    Float _learning_rate(learning_rate, PUBLIC);
    Float _n(n, PUBLIC);
    double grad_time_total = 0;
    double update_time_total = 0;
    double noise_time_total = 0;
    double aes_time_total = 0;
    emp::AES_128_CTR_Calculator aes_128_ctr_calculator = emp::AES_128_CTR_Calculator();

    // sample minibatch
    vector<vector<float>> x_tmp(samples.begin(), samples.end()); // randomly sample n
    vector<float> y_tmp(labels.begin(), labels.end());
    vector<vector<Float>> x_total;
    vector<Float> y_total;
    // convert to Float
    for (int i = 0; i < x_tmp.size(); ++i) {
        vector<Float> tmp;
        for (int j = 0; j < x_tmp[0].size(); ++j) tmp.push_back(Float(x_tmp[i][j], ALICE));
        x_total.push_back(tmp);
        y_total.push_back(Float(y_tmp[i], ALICE));
    }
    permute_samples(x_total, y_total, final_seed);

    for (int cur_step = 0; cur_step < steps; ++cur_step) {
        vector<vector<Float>> x(x_total.begin() + cur_step * k, x_total.begin() + (cur_step + 1) * k);
        vector<Float> y(y_total.begin() + cur_step * k, y_total.begin() + (cur_step + 1) * k);
        auto begin_grad = emp::clock_start();
        vector<Float> grad_sum = _grad(x, y, theta);
        grad_time_total += (time_from(begin_grad) / 1000.0);

        auto begin_aes = emp::clock_start();

        for (int i = 0; i < m / 4; ++i) {
            Integer ctr(128, i, PUBLIC);
            vector<Bit> ctr_bits = ctr.bits;
            aes_128_ctr_calculator.aes_128_ctr(&(final_seed[0].bit), &(final_seed[0].bit), &(ctr_bits[0].bit),
                                               nullptr, 128, PUBLIC, 0);
        }
        aes_time_total += (time_from(begin_aes) / 1000.0);

        auto begin_noise_1 = emp::clock_start();
        vector<Float> gaussian_noise;
        for (int i = 0; i < (int) (m / 4 + 1); ++i) {
            auto noise_pair = alt_gen_gaussian_noise(final_seed, 0, C * C * sigma * sigma, true);
            gaussian_noise.push_back(noise_pair.first);
            gaussian_noise.push_back(noise_pair.second);
            noise_pair = alt_gen_gaussian_noise(final_seed, 0, C * C * sigma * sigma, false);
            gaussian_noise.push_back(noise_pair.first);
            gaussian_noise.push_back(noise_pair.second);
        }
        noise_time_total += (time_from(begin_noise_1) / 1000.0);
        for (int i = 0; i < m; ++i) {
            auto begin_noise_2 = emp::clock_start();
            Float _z = gaussian_noise[i];
            auto begin_update = emp::clock_start();
            noise_time_total += (time_from(begin_noise_2) / 1000.0);
            Float update = (_learning_rate / _n) * (grad_sum[i] + _z);
            theta[i] = theta[i] - update;
            update_time_total += (time_from(begin_update) / 1000.0);
        }
    }
    if (party == ALICE) {
        cout << endl;
        cout << "total gradient time: " << grad_time_total << " ms\n";
        cout << "total update time: " << update_time_total << " ms\n";
        cout << "total noise time: " << (noise_time_total + aes_time_total) << " ms\n";
//        cout << "total aes time: " << aes_time_total / 1000.0 / steps << " s\n";
    }
}

void dp_ftrl(NetIO *io, int party, int steps, vector<vector<float>> samples,
             vector<float> labels, float learning_rate, float sigma, int k, float bounded_set_radius = 1) {
    int n = samples.size();    // "n" sample size
    int m = samples[0].size(); // "m" sample dimension
    // preparing seed
    block randomness_seed = get_common_randomness(io, party);
    bool seed_in_bits[128];
    block_to_bool(seed_in_bits, randomness_seed);
    vector<Bit> final_seed;
    final_seed.reserve(128);
    for (int i = 0; i < 128; ++i) {
        final_seed.push_back(Bit(seed_in_bits[i], ALICE));
    }

    vector<Float> theta(m, Float(0, PUBLIC));
//    permute_samples(samples, labels, final_seed);
    Float _learning_rate(learning_rate, PUBLIC);
    Float _n(n, PUBLIC);

    double grad_time_total = 0;
    double update_time_total = 0;
    double noise_time_total = 0;
    double aes_time_total = 0;
    double leaf_time_total = 0;
    emp::AES_128_CTR_Calculator aes_128_ctr_calculator = emp::AES_128_CTR_Calculator();
    for (int cur_step = 0; cur_step < steps; ++cur_step) {
//        permute_samples(samples, labels);
        // sample minibatch
        vector<vector<float>> x_tmp(samples.begin(), samples.begin() + n); // randomly sample n
        vector<float> y_tmp(labels.begin(), labels.begin() + n);
        vector<vector<Float>> x;
        vector<Float> y;
        // convert to Float

        for (int i = 0; i < n; ++i) {
            vector<Float> tmp;
            for (int j = 0; j < x_tmp[0].size(); ++j) tmp.push_back(Float(x_tmp[i][j], ALICE));
            x.push_back(tmp);
            y.push_back(Float(y_tmp[i], ALICE));
        }


        auto begin_aes = emp::clock_start();

        for (int i = 0; i < (int) ((2 * n - 1)) / 4 + 1; ++i) {
            Integer ctr(128, i, PUBLIC);
            vector<Bit> ctr_bits = ctr.bits;
            aes_128_ctr_calculator.aes_128_ctr(&(final_seed[0].bit), &(final_seed[0].bit), &(ctr_bits[0].bit),
                                               nullptr, 128, PUBLIC, 0);
        }
        aes_time_total += (time_from(begin_aes) / 1000.0);

        auto begin_noise_1 = emp::clock_start();
        // make tree
        vector<vector<Float>> the_tree;
        for (int i = 0; i < (int) ((2 * n - 1)); ++i) {
            vector<Float> noise_at_one_node;
            for (int j = 0; j < m / 4; ++j) {
                auto noise_pair = alt_gen_gaussian_noise(final_seed, 0, C * C * sigma * sigma, true);
                noise_at_one_node.push_back(noise_pair.first);
                noise_at_one_node.push_back(noise_pair.second);
                noise_pair = alt_gen_gaussian_noise(final_seed, 0, C * C * sigma * sigma, false);
                noise_at_one_node.push_back(noise_pair.first);
                noise_at_one_node.push_back(noise_pair.second);
            }
            the_tree.push_back(noise_at_one_node);
        }
        noise_time_total += (time_from(begin_noise_1) / 1000.0);


        for (int i = 0; i < n; ++i) {
            vector<vector<Float>> x_selected;
            vector<Float> y_selected;
            x_selected.push_back(x[i]);
            y_selected.push_back(y[i]);
            auto begin_grad = emp::clock_start();
            vector<Float> grad_sum = _grad(x_selected, y_selected, theta);
            grad_time_total += (time_from(begin_grad) / 1000.0);
            int leaf_idx = (2 * n - 1) - n + i;
            // udpate tree
            for (int j = 0; j < m; ++j) {
                the_tree[leaf_idx][j] = the_tree[leaf_idx][j] + grad_sum[j];
            }
            int parent_idx = leaf_idx;
            while (parent_idx != 0) {
                parent_idx = (int) floor((parent_idx - 1) / 2);
                for (int j = 0; j < m; ++j) {
                    the_tree[parent_idx][j] = the_tree[parent_idx][j] + grad_sum[j];
                }
            }
            auto begin_update = clock_start();
            for (int i = 0; i < m; ++i) {
                Float update = (_learning_rate) * (the_tree[0][i]);
                theta[i] = theta[i] - update;
            }
            update_time_total += (time_from(begin_update) / 1000.0);

        }


    }
    if (party == ALICE) {
        cout << endl;
        cout << "total gradient time: " << grad_time_total << " ms\n";
        cout << "total update time: " << update_time_total << " ms\n";
        cout << "total noise time: " << (noise_time_total + aes_time_total) << " ms\n";
//        cout << "total aes time: " << aes_time_total / 1000.0 / steps << " s\n";
    }
}

#endif //ZK_LR_LOG_REG_H
