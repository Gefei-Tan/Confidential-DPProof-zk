# Confidential DProof

Installation
=====

1. `wget https://raw.githubusercontent.com/emp-toolkit/emp-readme/master/scripts/install.py`
2. `python[3] install.py --deps --tool --ot --zk`
    1. By default it will build for Release. `-DCMAKE_BUILD_TYPE=[Release|Debug]` option is also available.
    2. No sudo?
       Change [`CMAKE_INSTALL_PREFIX`](https://cmake.org/cmake/help/v2.8.8/cmake.html#variable%3aCMAKE_INSTALL_PREFIX).
3. you might need to `cmake . -DCRYPTO_IN_CIRCUIT=true` for `emp-tool`. This is to enable the AES circuit
4. make a directory for OT-data `mkdir data`

Test
=====
To run a test:
`./run ./bin/[test_binary]`

For example, to run test for DP-FTRL:
`./run ./bin/test_bench_dp_ftrl`

To run test for DP-SGD:
`./run ./bin/test_benchmark`

Manually set the port:

run `./bin/[binary] 1 [port]` on the prover machine

run `./bin/[binary] 2 [port]` on the auditor machine


`[port]` should be the same. Offset of each party will be automatically adjusted.

