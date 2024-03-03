//
// Created by 77183 on 2024/3/1.
//

#ifndef YOLO_DEPLOY_CXX_CUDA_UTILS_H
#define YOLO_DEPLOY_CXX_CUDA_UTILS_H

#include <cuda_runtime_api.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif  // CUDA_CHECK

#if NV_TENSORRT_MAJOR >= 8
#define TRT_NOEXCEPT noexcept
#define TRT_CONST_ENQUEUE const
#else
#define TRT_NOEXCEPT
#define TRT_CONST_ENQUEUE
#endif

#endif //YOLO_DEPLOY_CXX_CUDA_UTILS_H
