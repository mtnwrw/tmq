//--------------------------------------------------------------------------------------------------
// TMQ                                                                          (c) TMQ Authors 2025
//--------------------------------------------------------------------------------------------------
// CUDA Helpers
// Creator: Martin Wawro
// SPDX-License-Identifier: MPL-2.0
//--------------------------------------------------------------------------------------------------

//-------------------------------------- Project  Headers ------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <exception>
#include <mutex>
#include <cuda_runtime.h>
#include <unordered_map>
#include <torch/extension.h>

#include <Python.h>

//-------------------------------------- Local Definitions -----------------------------------------

static bool FP16_WARNING_ISSUED=false;
static bool CONSTANT_MEMORY_INITIALIZED=false;
static std::mutex CONSTANT_MEMORY_LOCK;

// TODO (mw) maybe pass this in via __grid__constant__ params
__constant__ float ternaryPatterns[256*4];     // 4k constant memory, usual constant cache is 8k per SM

//------------------------------------- CUDA/Device Functions --------------------------------------

// TODO (mw) docs
__host__
inline static float sigmoid(float x) {
    float y = max(-3.0f, min(3.0f, x));
    float y2 = y*y;
    return y * (27.0f + y2) / (27.0f + 9.0f * y2);
}


// TODO (mw) docs
template<typename T>
__global__
void forwardSoftStepDerivativeKernelFP(const T * weights, T * output, float digamma, float delta, float scale, size_t size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        T x = weights[tid];
        T fx = floor(x);
        T y = max(-3.0f, min(3.0f, digamma * (x-fx) - digamma/2.0f));
        T y2 = y * y;
        T a = y2 - 9.f;
        T b = y2 + 3.f;
        T z = (a * a) / (9.f * b * b);
        output[tid] = scale * digamma * z / (2.0f * delta);
    }
}


// TODO (mw) docs
template<typename T, typename D>
__global__
void backwardSoftStepDerivativeKernelFP(const T * weights, const T * grad, D * output, float digamma, float delta, float scale, size_t size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        T x = weights[tid];
        T g = grad[tid];
        T fx = floor(x);
        T y = max(-3.0f, min(3.0f, digamma * (x-fx) - digamma/2.0f));
        T y2 = y * y;
        T a = y2 + 3.f;
        T z = 16.f * y * (y2 - 9.f) / (3.f * a * a * a);
        output[tid] = (D)(scale * digamma * digamma * (z / (2.0f * delta)) * g);
    }
}


template<typename T>
__global__
void forwardSoftStepKernelFP(const T * weights, T * output, float digamma, float delta, size_t size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        T x = weights[tid];
        T fx = floor(x);
        T y = max(-3.0f, min(3.0f, digamma * (x-fx) - digamma/2.0f));
        T y2 = y * y;
        T z = y * (27.0f + y2) / (27.0f + 9.0f * y2);
        output[tid] = z / (2.0f * delta) + 0.5 + fx;
    }
}


template<typename T, typename D>
__global__
void backwardSoftStepKernelFP(const T * weights, const T * grad, D * output, float digamma, float delta, float scale, size_t size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        T x = weights[tid];
        T g = grad[tid];
        T fx = floor(x);
        T y = max(-3.0f, min(3.0f, digamma * (x-fx) - digamma/2.0f));
        T y2 = y * y;
        T a = y2 - 9.f;
        T b = y2 + 3.f;
        T z = (a * a) / (9.f * b * b);
        output[tid] = (D)(scale * digamma * (z / (2.0f * delta)) * g);
    }
}


// format of coefficients: (kx*ky) x (channels)
// format of input tensor: (batch) x (pixels) x (channels)
// format of output tensor: (batch) x (channel) x (kernel) x (pixels)
// threads: BH * 1
// TODO (mw) docs
template<int BH, int BW, int NT>
__global__ void ternaryDWConvTransFP32(const float * inputTensor, float *outputTensor, const uint32_t * coeffs, int patches, int channels, int kernelSize) {
    constexpr static int BWM = BW+1;
    __shared__ float buffer[BH * BWM];
    static_assert(BW == 16);
    static_assert((NT % BW) == 0);
    const int channelstride = kernelSize * patches;
    const int basepatch = blockIdx.x * BH;
    const int tflatidx = threadIdx.x;
    const int chan = blockIdx.y * BW;
    {
        const int lpatch = tflatidx / BW;
        for (int y = 0; y < BH; y += NT / BW) {   // NT / BW = 4
            if (basepatch + lpatch + y < patches) {
                const float *inptr = inputTensor + (basepatch + lpatch + y) * channels;
                const int xs = chan + (tflatidx % BW);
                buffer[(lpatch + y) * BWM + (tflatidx % BW)] = (xs < channels) ? inptr[xs] : 0.f;
            } else buffer[(lpatch + y) * BWM + (tflatidx % BW)] = 0.f;
        }
    }
    __syncthreads();
    for (int inblock=0; inblock < kernelSize; inblock++) {
        if (basepatch + tflatidx < patches) {
            const int baseidx = tflatidx * BWM;
            int writeoffset = (chan * kernelSize + inblock) * patches + basepatch + tflatidx;
            uint32_t compact = coeffs[(chan / 16) + ((channels + 15) / 16) * inblock];
            compact = __byte_perm(compact, 0, 0x2103);
#pragma unroll
            for (int i = 0, cc = chan; i < 4; i++, cc += 4) {
                const uint32_t code = compact & 0xFF;
                const float *coeffptr = (const float *) ternaryPatterns + code * 4;
                if (cc < channels) outputTensor[writeoffset] = buffer[baseidx + i * 4 + 0] * coeffptr[0];
                writeoffset += channelstride;
                if (cc + 1 < channels) outputTensor[writeoffset] = buffer[baseidx + i * 4 + 1] * coeffptr[1];
                writeoffset += channelstride;
                if (cc + 2 < channels) outputTensor[writeoffset] = buffer[baseidx + i * 4 + 2] * coeffptr[2];
                writeoffset += channelstride;
                if (cc + 3 < channels) outputTensor[writeoffset] = buffer[baseidx + i * 4 + 3] * coeffptr[3];
                writeoffset += channelstride;
                compact = __byte_perm(compact, 0, 0x2103);
            }
        }
    }
    __syncthreads();
}



template<int BH, int BW, int NT>
__global__ void ternaryDWConvFP32(const float * inputTensor, float *outputTensor, const uint32_t * coeffs, int patches, int channels, int kernelSize) {
    __shared__ float buffer[BH * BW];
    float accu[BH/NT][BW] = {0};
    float unpacked[BW] = {0};
    const int chan = blockIdx.x * BW;
    const int tflatidx = threadIdx.x + threadIdx.y * blockDim.x;
    const int basepatch = blockIdx.y * BH;
#pragma unroll 1
    for (int inblock = 0; inblock < kernelSize; inblock++) {
        const int lpatch = tflatidx / BW;
        // ----------------------------------------------------
        // Read lhs data. Note that we do not use it multiple
        // times and the only reason we buffer it in shared mem
        // is to get coalescing while reading...
        // ----------------------------------------------------
        for (int y = 0; y < BH; y += NT / BW) {   // NT / BW = 4
            if (basepatch + lpatch + y < patches) {
                const float *inptr = inputTensor + (basepatch + lpatch + y) * channels * kernelSize + inblock * channels;
                const int xs = chan + (tflatidx % BW);
                buffer[(lpatch + y) * BW + (tflatidx % BW)] = (xs < channels) ? inptr[xs] : 0.f;
            } else buffer[(lpatch + y) * BW + (tflatidx % BW)] = 0.f;
        }
        // ----------------------------------------------------
        // Read und unpack coefficient block
        // ----------------------------------------------------
        uint32_t compact = coeffs[(chan / 16) + (channels + 15) / 16 * inblock];
        compact = __byte_perm(compact, 0, 0x2103);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint32_t code = compact & 0xFF;
            const float *coeffptr = (const float *) ternaryPatterns + code * 4;
            unpacked[i * 4 + 0] = coeffptr[0];
            unpacked[i * 4 + 1] = coeffptr[1];
            unpacked[i * 4 + 2] = coeffptr[2];
            unpacked[i * 4 + 3] = coeffptr[3];
            compact = __byte_perm(compact, 0, 0x2103);
        }
        __syncthreads();
        // ----------------------------------------------------
        // Do actual convolution...
        // ----------------------------------------------------
        const int cpatch = tflatidx * (BH/NT);
#pragma unroll
        for (int y = 0; y < (BH / NT); y++) {
#pragma unroll
            for (int x = 0; x < BW; x++) {
                accu[y][x] += buffer[(cpatch + y) * BW + x] * unpacked[x];
            }
        }
    }
    // ----------------------------------------------------
    // Reorg result into shared memory buffer to make use
    // of coalescing on global write-out...
    // ----------------------------------------------------
    __syncthreads();
    const int cpatch = tflatidx * (BH/NT);
    for (int y = 0; y < (BH / NT); y++) {
        for (int x = 0; x < BW; x++) buffer[(cpatch + y) * BW + x] = accu[y][x];
    }
    // ----------------------------------------------------
    // Wrtte back results
    // ----------------------------------------------------
    __syncthreads();
    for (int y = 0; y < BH; y += NT/BW) {
        const int lpatch = tflatidx / BW;
        if (basepatch + lpatch + y < patches) {
            float *outptr = outputTensor + (basepatch + lpatch + y) * channels;
            const int xs = chan + (tflatidx % BW);
            if (xs < channels) outptr[xs] = buffer[(lpatch + y) * BW + (tflatidx % BW)];
        }
    }
}



template<int BS>
__global__ void ternaryMVFP32(const float * inputVector, float * outputVector, const uint32_t * coeffs, int cRows, int cCols, int cStride) {
    // TODO (mw) optimize
    constexpr int COEFFBLOCK = 16;
    constexpr int WARPSIZE = 32;
    constexpr int HALFWARP = 16;
    static __shared__ float cache[BS];
    float accu[BS]={0};
    const int xbase = blockIdx.x * BS;
    const int tflatidx = threadIdx.x + blockDim.x * threadIdx.y;
    const int warpidx = tflatidx % WARPSIZE;
    const int cmaxcol = (cCols + COEFFBLOCK-1) / COEFFBLOCK;
    if (tflatidx < BS) cache[tflatidx] = 0.f;
    __syncthreads();
    for (int block=0; block < (cRows+blockDim.y-1) / blockDim.y; block++) {
        const int row = block * blockDim.y + threadIdx.y;
        const int xo = threadIdx.x * COEFFBLOCK;
        const int x = (xbase + xo) / COEFFBLOCK;
        float lhs = (row < cRows) ? inputVector[row] : 0.f;
        uint32_t compcoeffs = ((row < cRows) && (x < cmaxcol)) ? coeffs[row * cStride + (xbase + xo) / COEFFBLOCK] : 0;
        compcoeffs = __byte_perm(compcoeffs, 0, 0x2103);
#pragma unroll
        for (int i = 0; i < COEFFBLOCK; i += 4) {
            const uint32_t code = compcoeffs & 0xFF;
            const float *coeffptr = (const float *) ternaryPatterns + code * 4;
            accu[xo + i + 0] += lhs * coeffptr[0];
            accu[xo + i + 1] += lhs * coeffptr[1];
            accu[xo + i + 2] += lhs * coeffptr[2];
            accu[xo + i + 3] += lhs * coeffptr[3];
            compcoeffs = __byte_perm(compcoeffs, 0, 0x2103);
        }
    }
    for (int offset=HALFWARP; offset > 0 ; offset/=2) {
        for (int i = 0; i < BS; i++) {
            accu[i] += __shfl_down_sync(0xffffffff, accu[i], offset);
        }
    }
    if (warpidx == 0) {
        for (int i = 0; i < BS; i++) atomicAdd(&cache[i], accu[i]);
    }
    __syncthreads();
    if (tflatidx < BS) {
        if (blockIdx.x * BS + tflatidx < cCols) outputVector[blockIdx.x * BS + tflatidx] = cache[tflatidx];
    }
}


template<int BS, int TH, int TW, int NT, bool CLIP>
__global__
void ternaryMMFP32_4x8(const float * inputTensor, float *outputTensor, const uint32_t * coeffs, int iRows, int cRows, int cCols, int inStride, int cStride, int oStride) {
    // TODO (mw) optimize
    static __shared__ float incache[BS][BS];              // e.g. 16 kB for 64x64
    static __shared__ uint32_t coeffcache[BS][(BS/16)];   // e.g. 1 kB for 64x64
    float accu[TH][TW] = {0};                             // e.g. 32 registers for 4x8 threads
    const int tflatidx = threadIdx.x + threadIdx.y * blockDim.x;
    const int lastblock = (cRows + BS -1) / BS;
#pragma unroll 1
    for (int inblock=0; inblock < lastblock; inblock++) {
        // --------------------------------------------------------------------------------
        // Read LHS into shared memory cache, do on-the-fly transposition
        // --------------------------------------------------------------------------------
        {
            // FIXME (mw) uncoalesced global memory access, this should be done differently
            const int xbase = inblock * BS;
            const int ybase = blockIdx.y * BS;
            if constexpr ((BS * BS) / (TW * TH) == NT) {
                if constexpr(CLIP) {
                    for (int yout = 0; yout < BS; yout += BS / 2) {
                        const int myy = yout + (tflatidx % 32);
                        const int mybasex = 16 * (tflatidx / 32);
                        if (ybase + myy < iRows) {
                            auto  *inptr = (const float *) (inputTensor + mybasex + xbase + (ybase + myy) * inStride);
                            for (int myx = 0; myx < 16; myx++) {
                                incache[mybasex + myx][myy] = ((mybasex + xbase + myx) < cRows) ? inptr[myx] : 0.f;
                            }
                        } else {
                            for (int myx = 0; myx < 16; myx++) incache[mybasex + myx][myy] = 0.f;
                        }
                    }
                } else {
                    for (int yout = 0; yout < BS; yout += BS / 2) {
                        const int myy = yout + (tflatidx % 32);
                        const int mybasex = 16 * (tflatidx / 32);
                        const float4 *inptr = (const float4 *) (inputTensor + mybasex + xbase + (ybase + myy) * inStride);
#pragma unroll
                        for (int myx = 0; myx < 16; myx += 4) {
                            float4 tmp = *inptr++;
                            incache[mybasex + myx + 0][myy] = tmp.x;
                            incache[mybasex + myx + 1][myy] = tmp.y;
                            incache[mybasex + myx + 2][myy] = tmp.z;
                            incache[mybasex + myx + 3][myy] = tmp.w;
                        }
                    }
                }
            }
        }
        {
            const int colmax = (cCols + 15) / 16;
            const int xbase = (blockIdx.x * BS) / 16;
            const int ybase = inblock * BS;
            const uint32_t * coeffbase = coeffs + xbase + ybase * cStride;
            if constexpr (BS * BS / 16 == 2 * NT) {
                // to read: 64*4 entries
                if constexpr (CLIP) {
                    const int x = 2 * (tflatidx % (BS / 32));  // 0,1,0,1,0,1,0,1 -> 0,2,0,2,0,2,0,2
                    const int y = tflatidx / (BS / 32);
                    if (ybase + y < cRows) {
                        coeffcache[y][x] = (xbase+x < colmax) ? coeffbase[x + y * cStride] : 0;
                        coeffcache[y][x+1] = (xbase+x+1 < colmax) ? coeffbase[x + 1 + y * cStride] : 0;
                    } else {
                        *(uint2 *) &coeffcache[y][x] = make_uint2(0, 0);
                    }
                } else {
                    const int x = 2 * (tflatidx % (BS / 32));
                    const int y = tflatidx / (BS / 32);
                    uint2 tmp = *(uint2 *) (coeffbase + x + y * cStride);
                    *(uint2 *) &coeffcache[y][x] = tmp;
                }
            }
        }
        __syncthreads();

        const int xblock = (tflatidx / (BS/TW)) / 2;
        const int yblock = tflatidx % (BS/TH);
        const uint32_t iniperm = (xblock & 1) ? 0x0321 : 0x2103;
#pragma unroll 8
        for (int yi = 0; yi < BS; yi++) {
            const float4 inputregs = make_float4(incache[yi][yblock],           incache[yi][yblock+(BS/TH)],
                                                 incache[yi][yblock+(2*BS/TH)], incache[yi][yblock+(3*BS/TH)]);
            uint32_t compcoeffs = coeffcache[yi][xblock/2];
            compcoeffs = __byte_perm(compcoeffs, 0, iniperm);
#pragma unroll
            for (int xb = 0; xb < 2; xb++) {                        // adds two rows of (16) weights to the accumulator
                const uint32_t code = compcoeffs & 0xFF;
                const float *coeffptr = (const float *) ternaryPatterns + code * 4;
                accu[0][xb*4+0] += inputregs.x * coeffptr[0];
                accu[0][xb*4+1] += inputregs.x * coeffptr[1];
                accu[0][xb*4+2] += inputregs.x * coeffptr[2];
                accu[0][xb*4+3] += inputregs.x * coeffptr[3];
                accu[1][xb*4+0] += inputregs.y * coeffptr[0];
                accu[1][xb*4+1] += inputregs.y * coeffptr[1];
                accu[1][xb*4+2] += inputregs.y * coeffptr[2];
                accu[1][xb*4+3] += inputregs.y * coeffptr[3];
                accu[2][xb*4+0] += inputregs.z * coeffptr[0];
                accu[2][xb*4+1] += inputregs.z * coeffptr[1];
                accu[2][xb*4+2] += inputregs.z * coeffptr[2];
                accu[2][xb*4+3] += inputregs.z * coeffptr[3];
                accu[3][xb*4+0] += inputregs.w * coeffptr[0];
                accu[3][xb*4+1] += inputregs.w * coeffptr[1];
                accu[3][xb*4+2] += inputregs.w * coeffptr[2];
                accu[3][xb*4+3] += inputregs.w * coeffptr[3];
                compcoeffs = __byte_perm(compcoeffs, 0, 0x2103);
            }
        }
        __syncthreads();
    }
    // --------------------------------------------------------------------------------
    // Write out the results, this arrangement here is not really good as it does not
    // allow for coalescing
    // --------------------------------------------------------------------------------
    float * outptr = outputTensor + blockIdx.y * BS * cCols + blockIdx.x * BS;
    const int xout = tflatidx / (BS/TW) / 2;
    const int yout = tflatidx % (BS/TH);
    if constexpr(CLIP) {
        const int xbase = blockIdx.x * BS;
        const int ybase = blockIdx.y * BS;
        for (int y = 0; y < TH; y++) {
            if (ybase + yout + y * (BS / TH) < iRows) {
                float *outrow = (outptr + (yout + y * (BS / TH)) * oStride + xout * TW);
                for (int x = 0; x < TW; x++) {
                    if (xbase + xout * TW + x < cCols) *outrow++ = accu[y][x];
                }
            }
        }
    } else {
#pragma unroll
        for (int y = 0; y < TH; y++) {
            float4 *outrow = (float4 *) (outptr + (yout + y * (BS / TH)) * oStride + xout * TW);
#pragma unroll
            for (int x = 0; x < TW; x += 4) {
                *outrow = make_float4(accu[y][x], accu[y][x + 1], accu[y][x + 2], accu[y][x + 3]);
                outrow++;
            }
        }
    }
}

//---------------------------------------- Host Functions ------------------------------------------


/**
 * @brief Initialize CUDA constant memory with LUT for ternary coefficients
 *
 * This creates a lookup-table for a set of 4 ternary coefficients represented by an 8-bit number where the
 * two MSBs are the first coefficient and the least two bits represent the last coefficient. The following
 * lookup is used for each two-bit combination:
 *  - 00 -> 0
 *  - 01 -> 1
 *  - 10 -> -1
 *  - 11 -> 0 (technically undefined)
 *
 * The data is uploaded to the \c ternaryPatterns symbol and can be accessed by CUDA kernels as constant memory.
 * Memory requirements are \f$ 256 \times 16 = 4k \f$ bytes (each coefficient is represented by 32-bit FP, which should
 * be revised to 16-bit FP).
 */
static void constantInit() {
    CONSTANT_MEMORY_LOCK.lock();
    if (!CONSTANT_MEMORY_INITIALIZED) {
        float patterns[256*4];
        for (int p=0; p < 256; p++) {
            for (int i=0; i<4; i++) {
                int b = (p >> (6-2*i)) & 0x3;
                switch (b) {
                    case 0b00:
                        patterns[p*4+i] = 0.f;
                        break;
                    case 0b01:
                        patterns[p*4+i] = 1.f;
                        break;
                    case 0b10:
                        patterns[p*4+i] = -1.f;
                        break;
                    default:
                        patterns[p*4+i] = 0.f;
                        break;
                }
            }
        }
        cudaMemcpyToSymbol(ternaryPatterns, patterns, 256 * 4 * sizeof(float));
        CONSTANT_MEMORY_INITIALIZED = true;
    }
    CONSTANT_MEMORY_LOCK.unlock();
}


// TODO (mw) docs
torch::Tensor ternaryMMultCUDA(torch::Tensor lhs, torch::Tensor weights, int columns) {
    constexpr int BS = 64;
    TORCH_CHECK(lhs.device().type() == torch::kCUDA);
    TORCH_CHECK(weights.device().type() == torch::kCUDA);
    TORCH_CHECK(lhs.dtype() == torch::kFloat32);
    TORCH_CHECK(weights.sizes().size() == 2);
    TORCH_CHECK(weights.dtype() == torch::kInt32);              // NOTE (mw) we cannot use uint32 as torch.save() cannot handle it
    constantInit();
    int lrows, lcols, batch=1;
    auto lshape = lhs.sizes();
    auto wsize = weights.sizes();
    TORCH_CHECK(wsize[1] * 16 >= columns);
    TORCH_CHECK(wsize[1] * 16 < columns + 16);
    int wrows = wsize[0];
    auto options = torch::TensorOptions().device(weights.device()).dtype(lhs.dtype());
    torch::Tensor output;
    switch (lshape.size()) {
        case 1:
          TORCH_CHECK(wsize[1]*16 >= columns);
          lrows = 1;
          lcols = lshape[0];
          output = torch::empty({lrows, columns}, options);
          break;
        case 2:
          TORCH_CHECK(wsize[1]*16 >= columns);
          lrows = lshape[0];
          lcols = lshape[1];
          output = torch::empty({lrows,columns}, options);
          break;
        case 3:
          TORCH_CHECK(wsize[1]*16 >= columns);
          batch = lshape[0];
          lrows = lshape[1];
          lcols = lshape[2];
          output = torch::empty({batch, lrows, columns}, options);
          break;
        case 4:
          TORCH_CHECK(wsize[1]*16 >= columns);
          TORCH_CHECK(lshape[0] == 1);
          batch = lshape[1];
          lrows = lshape[2];
          lcols = lshape[3];
          output = torch::empty({1,batch,lrows,columns}, options);
          break;
    }
    bool clip = ((lrows % BS) != 0) || ((lcols % BS) != 0) || ((columns % BS) != 0);
    int xblocks = (columns+BS-1)/BS;
    int yblocks = (lrows+BS-1)/BS;
    int cstride = wsize[1];
    if (!clip) {
        // TODO (mw) use a different MM algorithm for batched operation for efficiency
        float * lhsdata = lhs.data_ptr<float>();
        float * outdata = output.data_ptr<float>();
        for (int b=0; b < batch; b++) {
            ternaryMMFP32_4x8<BS, 4, 8, 128, false><<<dim3(xblocks, yblocks, 1), dim3(128, 1, 1)>>>(lhsdata, outdata, (const uint32_t *)weights.data_ptr<int32_t>(),
                                                                                                    lrows, wrows, columns, lcols, cstride, columns);
            lhsdata += lrows * lcols;
            outdata += lrows * columns;
        }
    } else {
        // TODO (mw) use a different MM algorithm for batched operation for efficiency
        float * lhsdata = lhs.data_ptr<float>();
        float * outdata = output.data_ptr<float>();
        for (int b=0; b < batch; b++) {
            ternaryMMFP32_4x8<BS, 4, 8, 128, true><<<dim3(xblocks, yblocks, 1), dim3(128, 1, 1)>>>(lhsdata, outdata, (const uint32_t *)weights.data_ptr<int32_t>(),
                                                                                                    lrows, wrows, columns, lcols, cstride, columns);
            lhsdata += lrows * lcols;
            outdata += lrows * columns;
        }
    }
    return output;
}


/**
 * @brief Perform ternary matrix/vector multiplication using CUDA
 *
 * @param vector
 * @param weights
 * @param columns Actual number of columns in original (non-transposed) weight matrix
 *
 * @return
 *
 */
// TODO (mw) docs
torch::Tensor ternaryMVMultCUDA(torch::Tensor vector, torch::Tensor weights, int columns) {
    constexpr int BS = 16;
    constexpr int COEFFPACK = 16;
    TORCH_CHECK(vector.device().type() == torch::kCUDA);
    TORCH_CHECK(weights.device().type() == torch::kCUDA);
    TORCH_CHECK(vector.dtype() == torch::kFloat32);
    TORCH_CHECK(vector.sizes().size() >= 1);
    TORCH_CHECK(weights.sizes().size() == 2);
    constantInit();
    auto vshape = vector.sizes();
    auto wsize = weights.sizes();
    TORCH_CHECK(wsize[1]*16 >= columns);
    TORCH_CHECK(wsize[1]*16 < columns+16);
    auto options = torch::TensorOptions().device(vector.device()).dtype(vector.dtype());
    int rows = wsize[0];
    torch::Tensor output;
    switch (vshape.size()) {
        case 1:
          output = torch::empty({columns}, options);
          break;
        case 2:
          TORCH_CHECK(vshape[0] == 1);
          output = torch::empty({1,columns}, options);
          break;
        case 3:
          TORCH_CHECK(vshape[0] == 1);
          TORCH_CHECK(vshape[1] == 1);
          output = torch::empty({1,1,columns}, options);
          break;
        case 4:
          TORCH_CHECK(vshape[0] == 1);                       // FIXME (mw) support batched operation here ?
          TORCH_CHECK(vshape[1] == 1);
          TORCH_CHECK(vshape[2] == 1);
          output = torch::empty({1,1,1,columns}, options); // FIXME (mw) support batched operation here ?
          break;
    }
    int blocks = (columns+BS-1) / BS;
    int ythreads = std::min(256, rows);
    ternaryMVFP32<BS><<<dim3(blocks,1,1), dim3(1,ythreads,1)>>>(vector.data_ptr<float>(), output.data_ptr<float>(), (const uint32_t *)weights.data_ptr<int32_t>(), rows, columns, wsize[1]);
    return output;
}


// TODO (mw) docs
torch::Tensor ternaryDWConvCUDA(torch::Tensor input, torch::Tensor weights, int kx, int ky, int channels) {
    constexpr int BW = 16;
    constexpr int BH = 128;
    constexpr int NT = 64;
    TORCH_CHECK(input.device().type() == torch::kCUDA);
    TORCH_CHECK(weights.device().type() == torch::kCUDA);
    TORCH_CHECK(input.sizes().size() > 1);
    constantInit();
    auto shape = input.sizes();
    int bs = 1, patches, kchan;
    if (shape.size() == 4) {
        bs = shape[0];
        TORCH_CHECK(shape[1] == 1);
        patches = shape[2];
        kchan = shape[3];
    }
    if (shape.size() == 3) {
        bs = shape[0];
        patches = shape[1];
        kchan = shape[2];
    }
    if (shape.size() == 2) {
        patches = shape[0];
        kchan = shape[1];
    }
    TORCH_CHECK(kchan == channels * kx * ky);
    auto options = torch::TensorOptions().device(input.device()).dtype(input.dtype());
    torch::Tensor output;
    if (shape.size() == 4) output = torch::empty({bs, 1, patches, channels}, options);
    if (shape.size() == 3) output = torch::empty({bs, patches, channels}, options);
    else output = torch::empty({patches, channels}, options);
    int xblocks = (channels + BW-1)/BW;
    int yblocks = (patches + BH-1)/BH;
    ternaryDWConvFP32<BH, BW, NT><<<dim3(xblocks, yblocks, 1), dim3(NT,1,1)>>>(input.data_ptr<float>(), output.data_ptr<float>(), (const uint32_t *)weights.data_ptr<int32_t>(), patches, channels, kx*ky);
    return output;
}


torch::Tensor ternaryDWConvTransCUDA(torch::Tensor input, torch::Tensor weights, int kx, int ky) {
    constexpr int BW = 16;
    constexpr int BH = 128;
    TORCH_CHECK(input.device().type() == torch::kCUDA);
    TORCH_CHECK(weights.device().type() == torch::kCUDA);
    TORCH_CHECK(input.sizes().size() > 1);
    TORCH_CHECK(weights.is_contiguous());
    constantInit();
    auto shape = input.sizes();
    int bs = 1, patches, chan;
    if (shape.size() == 4) {
        bs = shape[0];
        TORCH_CHECK(shape[1] == 1);
        patches = shape[2];
        chan = shape[3];
    }
    if (shape.size() == 3) {
        bs = shape[0];
        patches = shape[1];
        chan = shape[2];
    }
    if (shape.size() == 2) {
        patches = shape[0];
        chan = shape[1];
    }
    auto wshape = weights.sizes();
    TORCH_CHECK(wshape.size() == 4);
    TORCH_CHECK(wshape[0] == 1);
    TORCH_CHECK(wshape[1] == ky);
    TORCH_CHECK(wshape[2] == kx);
    TORCH_CHECK(wshape[3] == (chan+15)/16);
    int ksize = kx * ky;
    auto options = torch::TensorOptions().device(input.device()).dtype(input.dtype());
    torch::Tensor output;
    if (shape.size() == 4) output = torch::empty({bs, 1, chan * ksize, patches}, options);
    if (shape.size() == 3) output = torch::empty({bs, chan * ksize, patches}, options);
    else output = torch::empty({chan * ksize, patches}, options);
    int xblocks = (patches + BH-1) / BH;
    int yblocks = (chan + BW-1) / BW;
    ternaryDWConvTransFP32<BH, BW, BH><<<dim3(xblocks,yblocks,1), dim3(BH,1,1)>>>(input.data_ptr<float>(), output.data_ptr<float>(), (const uint32_t *)weights.data_ptr<int32_t>(), patches, chan, kx*ky);
    return output;
}


// TODO (mw) docs
torch::Tensor forwardSoftStepDerivativeCUDA(torch::Tensor weights, float digamma, float strength) {
    TORCH_CHECK(weights.device().type() == torch::kCUDA);
    auto options = torch::TensorOptions().device(weights.device()).dtype(weights.dtype());
    auto output = torch::empty(weights.sizes(), options);
    float g = max(1e-2f, digamma);
    if (weights.dtype() == torch::kFloat32) {
        int mingrid, minblock;
        size_t size = weights.numel();
        auto rc = cudaOccupancyMaxPotentialBlockSize(&mingrid, &minblock, forwardSoftStepDerivativeKernelFP<float>, 0, 0);
        TORCH_CHECK(rc == cudaSuccess);
        int blocks = (size + minblock - 1) / minblock;
        if (blocks == 1) minblock = size;
        forwardSoftStepDerivativeKernelFP<float><<<dim3(blocks,1,1), dim3(minblock, 1, 1)>>>(weights.data_ptr<float>(), output.data_ptr<float>(), g, sigmoid(g/2.0f), strength, size);
    } else if (weights.dtype() == torch::kFloat16) {
        throw std::invalid_argument("Not implemented yet");
    } else {
        throw std::invalid_argument("Expected either a float16 or a float32 tensor as input/output");
    }
    return output;
}


// TODO (mw) docs
torch::Tensor backwardSoftStepDerivativeCUDA(torch::Tensor weights, float digamma, float scale, torch::Tensor grad) {
    TORCH_CHECK(weights.device().type() == torch::kCUDA);
    TORCH_CHECK(grad.device().type() == torch::kCUDA);
    auto options = torch::TensorOptions().device(weights.device()).dtype(weights.dtype());
    torch::Tensor output = torch::empty(weights.sizes(), options);
    float g = max(1e-2f, digamma);
    if (weights.dtype() == torch::kFloat32) {
        TORCH_CHECK(weights.dtype() == grad.dtype());
        int mingrid, minblock;
        size_t size = weights.numel();
        auto rc = cudaOccupancyMaxPotentialBlockSize(&mingrid, &minblock, backwardSoftStepDerivativeKernelFP<float,float>, 0, 0);
        TORCH_CHECK(rc == cudaSuccess);
        int blocks = (size + minblock - 1) / minblock;
        if (blocks == 1) minblock = size;
        backwardSoftStepDerivativeKernelFP<float,float><<<dim3(blocks,1,1), dim3(minblock, 1, 1)>>>(weights.data_ptr<float>(), grad.data_ptr<float>(), output.data_ptr<float>(), g, sigmoid(g/2.0f), scale, size);
    } else if (weights.dtype() == torch::kFloat16) {
        // TODO (mw) add code here
        throw std::invalid_argument("Not implemented yet");
    } else {
        throw std::invalid_argument("Expected either a float16 or a float32 tensor as input/output");
    }
    return output;
}



// TODO (mw) docs
torch::Tensor forwardSoftStepCUDA(torch::Tensor weights, float digamma) {
    TORCH_CHECK(weights.device().type() == torch::kCUDA);
    auto options = torch::TensorOptions().device(weights.device()).dtype(weights.dtype());
    auto output = torch::empty(weights.sizes(), options);
    float g = max(1e-2f, digamma);
    if (weights.dtype() == torch::kFloat32) {
        int mingrid, minblock;
        size_t size = weights.numel();
        auto rc = cudaOccupancyMaxPotentialBlockSize(&mingrid, &minblock, forwardSoftStepKernelFP<float>, 0, 0);
        TORCH_CHECK(rc == cudaSuccess);
        int blocks = (size + minblock - 1) / minblock;
        if (blocks == 1) minblock = size;
        forwardSoftStepKernelFP<float><<<dim3(blocks,1,1), dim3(minblock, 1, 1)>>>(weights.data_ptr<float>(), output.data_ptr<float>(), g, sigmoid(g/2.0f), size);
    } else if (weights.dtype() == torch::kFloat16) {
        if (!FP16_WARNING_ISSUED) {
            PySys_WriteStderr("It is not recommended to use FP16 with quantization during training, as overflows may occur\n");
            FP16_WARNING_ISSUED=true;
        }
        // TODO (mw) add code here
        throw std::invalid_argument("Not implemented yet");
    } else {
        throw std::invalid_argument("Expected either a float16 or a float32 tensor as input/output");
    }
    return output;
}


// TODO (mw) docs
torch::Tensor backwardSoftStepCUDA(torch::Tensor weights, float digamma, float scale, torch::Tensor grad) {
    TORCH_CHECK(weights.device().type() == torch::kCUDA);
    TORCH_CHECK(grad.device().type() == torch::kCUDA);
    auto options = torch::TensorOptions().device(weights.device()).dtype(weights.dtype());
    torch::Tensor output = torch::empty(weights.sizes(), options);
    float g = max(1e-2f, digamma);
    if (weights.dtype() == torch::kFloat32) {
        TORCH_CHECK(weights.dtype() == grad.dtype());
        int mingrid, minblock;
        size_t size = weights.numel();
        auto rc = cudaOccupancyMaxPotentialBlockSize(&mingrid, &minblock, backwardSoftStepKernelFP<float,float>, 0, 0);
        TORCH_CHECK(rc == cudaSuccess);
        int blocks = (size + minblock - 1) / minblock;
        if (blocks == 1) minblock = size;
        backwardSoftStepKernelFP<float,float><<<dim3(blocks,1,1), dim3(minblock, 1, 1)>>>(weights.data_ptr<float>(), grad.data_ptr<float>(), output.data_ptr<float>(), g, sigmoid(g/2.0f), scale, size);
    } else if (weights.dtype() == torch::kFloat16) {
        if (!FP16_WARNING_ISSUED) {
            PySys_WriteStderr("It is not recommended to use FP16 with quantization during training, as overflows may occur\n");
            FP16_WARNING_ISSUED=true;
        }
        // TODO (mw) add code here
        throw std::invalid_argument("Not implemented yet");
    } else {
        throw std::invalid_argument("Expected either a float16 or a float32 tensor as input/output");
    }
    return output;
}


//---------------------------------------- Python Bindings -----------------------------------------


PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("ternary_mmm_cuda", &ternaryMMultCUDA, "Ternary weights matrix/matrix multiplication")
       .def("ternary_mvm_cuda", &ternaryMVMultCUDA, "Ternary weights matrix/vector multiplication")
       .def("fwd_softstep", &forwardSoftStepCUDA, "Forward mapping for the soft-step function")
       .def("bwd_softstep", &backwardSoftStepCUDA, "Backward mapping for the soft-step function (including gradient multiplication)")
       .def("fwd_softstep_derivative", &forwardSoftStepDerivativeCUDA, "Forward mapping for the derivative of the soft-step function")
       .def("bwd_softstep_derivative", &backwardSoftStepDerivativeCUDA, "Backward mapping for the derivative of the soft-step function (including gradient multiplication)")
       .def("ternary_dwconv_cuda", &ternaryDWConvCUDA, "Ternary depthwise convolution")
       .def("ternary_dwconvtrans_cuda", &ternaryDWConvTransCUDA, "Ternary depthwise transpose convolution");
}
