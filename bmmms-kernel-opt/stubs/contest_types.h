// 与 npu-v1/project/main.asc 里定义的 ABI 结构一致（main.asc 先定义、再 include kernel.asc）
#pragma once
#include <cstdint>
struct TensorInfo {
    const int64_t* shape;
    int64_t numDims;
    int32_t dtype;
};
struct TensorGroupInfo {
    const TensorInfo* tensors;
    int64_t numTensors;
};
