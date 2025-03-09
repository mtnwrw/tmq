//--------------------------------------------------------------------------------------------------
// TMQ                                                                          (c) TMQ Authors 2025
//--------------------------------------------------------------------------------------------------
// Native Helpers
// Creator: Martin Wawro
// SPDX-License-Identifier: MPL-2.0
//--------------------------------------------------------------------------------------------------

//-------------------------------------- Project  Headers ------------------------------------------

#include "rangecoder.h"

//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <pybind11/pybind11.h>
#ifndef WIN32
#include <arpa/inet.h>
#else
#include <winsock2.h>
#endif
#include <torch/extension.h>

//-------------------------------------- Local Definitions -----------------------------------------

static uint64_t htonll(uint64_t value) {
    if (__BYTE_ORDER == __LITTLE_ENDIAN) return (((uint64_t)htonl(value & 0xFFFFFFFF)) << 32) | htonl(value >> 32);
    else return value;
}

static uint64_t ntohll(uint64_t value) {
    if (__BYTE_ORDER == __LITTLE_ENDIAN) return (((uint64_t)ntohl(value & 0xFFFFFFFF)) << 32) | ntohl(value >> 32);
    else return value;
}


/**
 * @brief Run entropy coding (simple range coding) on 2D bit-stuffed matrix
 *
 * @param input Input matrix to compress, stored in compact 2-bit per entry format
 * @param rows Number of rows in the matrix
 * @param columns Number of columns in the matrix
 *
 * @return Combination of output data-stream and size of the data-stream
 */
static std::pair<uint8_t *, uint32_t> entropyEncodeTernary(const uint32_t * input, int64_t rows, int64_t columns) {
    uint8_t symtable[4] = {1, 2, 0, 3};   // 0, -1, 1, _
    int64_t histo[4] = {0};               // -1, 0, 1, _
    int qcols = (int)((columns+15)/16);
    for (int64_t r=0; r < rows; r++) {
        for (int qc=0; qc < qcols; qc++) {
            uint32_t shiftreg = input[r*qcols+qc];
            int lim = (int)std::min((int64_t)16, columns - (qc*16));
            for (int i=0; i < lim; i++) {
                uint8_t bits = (int) ((shiftreg >> (2 * (15 - i))) & 3);
                histo[symtable[bits]]++;
            }
        }
    }
    if (histo[3] != 0) throw std::runtime_error("Illegal data encountered");
    int64_t total = histo[0] + histo[1] + histo[2];
    float scale = 65535.f / (float)total;
    std::vector<uint16_t> probs;
    for (int i=0; i<3; i++) probs.push_back(std::max((uint16_t)0, std::min((uint16_t)65535, (uint16_t)((float)histo[i] * scale))));
    RangeEncoder<uint8_t> coder(probs);
    for (int64_t r=0; r < rows; r++) {
        for (int qc=0; qc < qcols; qc++) {
            uint32_t shiftreg = input[r*qcols+qc];
            int lim = (int)std::min((int64_t)16, columns - (qc*16));
            for (int i=0; i < lim; i++) {
                uint8_t bits = (int) ((shiftreg >> (2 * (15 - i))) & 3);
                coder.encode(symtable[bits]);
            }
        }
    }
    coder.flush();
    size_t bytes = coder.stream().numBytes();
    size_t totalsize = bytes + 8 + 8 + 4 + 6;
    int64_t numsyms = rows * columns;
    if (totalsize > (((size_t)1)<<32)-1) throw std::invalid_argument("Data too large");
    uint8_t * output = new uint8_t[totalsize];
    *((uint64_t *)output) = htonll((uint64_t)numsyms);
    *((uint64_t *)(output+8)) = htonll((uint64_t)coder.stream().numBits());
    *((uint32_t *)(output+8+8)) = htonl((uint32_t)totalsize);
    {
        uint16_t * probtbl = (uint16_t *)(output+8+8+4);
        for (int i=0; i <3; i++) probtbl[i] = htons(probs[i]);
    }
    if (coder.stream().numBytes() > 0) {
        memcpy(output+8+8+4+6, &(coder.stream().buffer()[0]), coder.stream().numBytes());
    }
    return {output, (uint32_t)totalsize};
}


/**
 * @brief Run entropy decoding on a range-coded ternary matrix into bit-stuffed target matrix
 *
 * @param input Pointer to entropy-coded input datastream
 * @param[out] output Pointer to 2-bit bit-stuffed (2 -> 32) matrix as result
 * @param rows Rows in the matrix to decompress
 * @param cols Columns (not bit-stuffed columns)of the matrix to decompress
 *
 * Decompress the entropy-coded datastream to a 2-bit compact bit-stuffed representation that stores 16 elemnets
 * in a 32-bit word.
 */
static uint64_t entropyDecodeTernary(const uint8_t *input, uint32_t *output, int64_t rows, int64_t columns) {
    uint8_t symtable[3] = {2, 0, 1};
    uint64_t numsyms = ntohll(*(uint64_t *)input);
    if (numsyms == 0) {
        *output = 0;
        return numsyms;
    }
    uint64_t bits = ntohll(*(uint64_t *)(input+8));
    uint32_t totalsize = ntohl(*(uint32_t *)(input+8+8));
    TORCH_CHECK(totalsize >= 8+8+4+6);
    std::vector<uint16_t> probs;
    for (int i=0; i<3; i++) probs.push_back(ntohs(*(uint16_t *)(input+8+8+4+2*i)));
    ReadStream<const uint8_t *> stream(input+8+8+4+6, (size_t)bits);
    RangeDecoder<uint8_t, ReadStream<const uint8_t *>> decoder(probs, stream);
    int qcols = (int)((columns+15)/16);
    for (int64_t r=0; r < rows; r++) {
        for (int qc=0; qc < qcols; qc++) {
            uint32_t shiftreg = 0;
            int lim = (int)std::min((int64_t)16, columns - (qc*16));
            for (int i=0; i < lim; i++) {
                uint8_t sym = decoder.decode();
                uint8_t bits = symtable[sym];
                shiftreg = (shiftreg << 2) | bits;
            }
            if (lim != 16) shiftreg <<= (2 * (16-lim));
            *output++ = shiftreg;
        }
    }
    return numsyms;
}


/**
 * @brief Run 2-bit bit-stuffing on a 2D matrix
 *
 * @param input Pointer to input matrix data in row-major order
 * @param[out] output Pointer to output data, formatted like \c [in*kx*ky][ceil(out/16)]
 * @param rows Number of output channels
 * @param columns Number of input channels
 *
 */
template<typename T>
static void matrixTo2BitTernary(const T *input, uint32_t *output, int64_t rows, int64_t columns) {
    int64_t qcols = (columns+15)/16;
    for (int64_t r=0; r < rows; r++) {
        for (int64_t c=0; c < columns; c += 16) {
            uint32_t shiftreg = 0;
            for (int i=0; i < 16; i++) {
                int n=0;
                if (c+i < columns) {
                    if constexpr (std::is_same<float, T>()) n = (int)roundf(input[r*columns+c+i]);
                    else n = (int)input[r*columns+c+i];
                }
                n = std::min(1, std::max(-1, n));
                int bits = (n==0) ? 0 : ((n==1) ? 1 : 2);       // 0 -> 0, 1 -> 1, -1 -> 2
                shiftreg = (shiftreg << 2) | bits;
            }
            output[r*qcols + (c/16)] = shiftreg;
        }
    }
}


/**
 * @brief Run 2-bit bit-stuffing on a 4D convolution kernel
 *
 * @param input Pointer to input data, formatted like \c [out][in][ky][kx]
 * @param[out] output Pointer to output data, formatted like \c [in*kx*ky][ceil(out/16)]
 * @param out Number of output channels
 * @param in Number of input channels
 * @param ky Kernel size (y-direction)
 * @param kx Kernel size (x-direction)
 *
 */
template<typename T>
static void kernelTo2BitTernary(const T *input, uint32_t *output, int64_t out, int64_t in, int64_t ky, int64_t kx) {
    // NOTE (mw) we store the data in transposed form as the ternary CUDA GEMM kernel that is using it later
    // requires the bit-stuffing along each row
    int64_t outchanstride = kx * ky * in;
    int64_t qoutchannels = (out + 15) / 16;
    for (int64_t row=0; row < in * kx * ky; row++) {
        for (int64_t qout=0; qout < qoutchannels; qout++) {
            uint32_t shiftreg = 0;
            for (int i=0; i < 16; i++) {
                int n = 0;
                int64_t outchannel = qout * 16 + i;
                if (outchannel < out) {
                    if constexpr (std::is_same<float, T>()) n = (int)roundf(input[row + outchannel * outchanstride]);
                    else n = (int)input[row + outchannel * outchanstride];
                }
                n = std::min(1, std::max(-1, n));
                int bits = (n==0) ? 0 : ((n==1) ? 1 : 2);       // 0 -> 0, 1 -> 1, -1 -> 2
                shiftreg = (shiftreg << 2) | bits;
            }
            output[row * qoutchannels + qout] = shiftreg;
        }
    }
}


/**
 * @brief
 *
 * @param input
 * @param output
 * @param rows
 * @param columns
 *
 */
static void compactMatrixTo32BitFloat(const uint32_t *input, float *output, int64_t rows, int64_t columns) {
    int64_t qcols = (columns+15) / 16;
    for (int64_t r=0; r < rows; r++) {
        for (int64_t c=0; c < qcols; c++) {
            uint32_t shiftreg = input[r*qcols+c];
            for (int i=0; i < 16; i++) {
                if (c*16 + i < columns) {
                    int bits = (int)((shiftreg >> (2 * (15-i))) & 3);
                    float val = (bits == 0) ? 0.f : ((bits == 1) ? 1.0f : -1.0f);
                    output[r * columns + c*16+i] = val;
                }
            }
        }
    }
}



/**
 * @brief
 *
 * @param input
 * @param output
 * @param out
 * @param in
 * @param ky
 * @param kx
 *
 */
static void compactKernelTo32BitFloat(const uint32_t *input, float *output, int64_t out, int64_t in, int64_t ky, int64_t kx) {
    // NOTE (mw) the compact representation stores the data in transposed form, as the ternary CUDA
    // GEMM kernel that is using it later requires the bit-stuffing along each row, this is reversed here
    int64_t outchanstride = kx * ky * in;
    int64_t qoutchannels = (out + 15) / 16;
    for (int64_t row=0; row < in * kx * ky; row++) {
        for (int64_t qout = 0; qout < qoutchannels; qout++) {
            uint32_t shiftreg = input[row * qoutchannels + qout];
            for (int64_t i = 0; i < 16; i++) {
                if (qout * 16 + i < out) {
                    int bits = (int) ((shiftreg >> (2 * (15 - i))) & 3);
                    float val = (bits == 0) ? 0.f : ((bits == 1) ? 1.0f : -1.0f);
                    output[row + (qout * 16 + i) * outchanstride] = val;
                }
            }
        }
    }
}




/**
 * @brief
 *
 * @param input
 *
 * @return
 *
 */
static torch::Tensor compactTernaryConvKernel(torch::Tensor input) {
    auto shape = input.sizes();
    int64_t out = shape[0];
    int64_t in = shape[1];
    int64_t ky = shape[2];
    int64_t kx = shape[3];
    int64_t qout = (out+15)/16;
    // NOTE (mw) we are going to store the weights in int32 blocks because torch does not support saving uint32
    auto options = torch::TensorOptions().device(input.device()).dtype(torch::kInt32);
    auto output = torch::empty({in, ky, kx, qout}, options);
    if (input.dtype() == torch::kFloat32) kernelTo2BitTernary(input.data_ptr<float>(), (uint32_t *)output.data_ptr<int32_t>(), out, in, ky, kx);
    else if (input.dtype() == torch::kInt32) kernelTo2BitTernary(input.data_ptr<int32_t>(), (uint32_t *)output.data_ptr<int32_t>(), out, in, ky, kx);
    else if (input.dtype() == torch::kInt8) kernelTo2BitTernary(input.data_ptr<int8_t>(), (uint32_t *)output.data_ptr<int32_t>(), out, in, ky, kx);
    else throw std::invalid_argument("Cannot handle the supplied data-type");
    return output;
}



/**
 *
 * This function stores the \b transposed input matrix which is assumed to only consist of ternary values (-1,0,1)
 * by performing bit-stuffing into 32-bit words using 2 bits per element.
 */
static torch::Tensor compactTernaryMatrix(torch::Tensor input) {
    auto shape = input.sizes();
    TORCH_CHECK(shape.size() > 1);
    int64_t rows = shape[0];
    int64_t cols = shape[1];
    // NOTE (mw) we are going to store the weights in int32 blocks because torch does not support saving uint32
    auto options = torch::TensorOptions().device(input.device()).dtype(torch::kInt32);
    // NOTE (mw) we transpose here
    auto tmp = input.transpose(0,1).contiguous();
    int64_t qcols = (rows+15)/16;
    auto output = torch::empty({cols,qcols}, options);
    if (input.dtype() == torch::kFloat32) matrixTo2BitTernary(tmp.data_ptr<float>(), (uint32_t *)output.data_ptr<int32_t>(), cols, rows);
    else if (input.dtype() == torch::kInt32) matrixTo2BitTernary(tmp.data_ptr<int32_t>(), (uint32_t *)output.data_ptr<int32_t>(), cols, rows);
    else if (input.dtype() == torch::kInt8) matrixTo2BitTernary(tmp.data_ptr<int8_t>(), (uint32_t *)output.data_ptr<int32_t>(), cols, rows);
    else throw std::invalid_argument("Cannot handle the supplied data-type");
    return output;
}


/**
 * @brief
 *
 * @param input
 *
 * @return
 */
static torch::Tensor compressTernaryMatrix(torch::Tensor& input, bool fromCompact) {
    auto shape = input.sizes();
    int64_t rows = shape[0];
    int64_t cols = shape[1];
    // NOTE (mw) we transpose here
    auto tmp = input.transpose(0,1).contiguous();
    int64_t qcols = (rows+15)/16;
    uint32_t * compact;
    if (fromCompact) compact = (uint32_t *)input.data_ptr<int32_t>();
    else {
        compact = new uint32_t[cols * qcols];
        if (input.dtype() == torch::kFloat32) matrixTo2BitTernary(tmp.data_ptr<float>(), compact, cols, rows);
        else if (input.dtype() == torch::kInt32) matrixTo2BitTernary(tmp.data_ptr<int32_t>(), compact, cols, rows);
        else if (input.dtype() == torch::kInt8) matrixTo2BitTernary(tmp.data_ptr<int8_t>(), compact, cols, rows);
        else {
            delete [] compact;
            throw std::invalid_argument("Cannot handle the supplied data-type");
        }
    }
    auto eres = entropyEncodeTernary(compact, cols, rows);
    if (!fromCompact) delete [] compact;
    if (eres.first == nullptr || eres.second == 0) {
        delete [] eres.first;
        throw std::runtime_error("Cannot encode data");
    }
    auto options = torch::TensorOptions().device(input.device()).dtype(torch::kInt8);
    auto output = torch::empty(eres.second, options);
    memcpy(output.data_ptr<int8_t>(), eres.first, eres.second);
    delete [] eres.first;
    return output;
}



/**
 * @brief
 *
 * @param input
 * @param fromCompact
 *
 * @return
 */
static torch::Tensor compressTernaryConvKernel(torch::Tensor& input, bool fromCompact) {
    auto shape = input.sizes();
    int64_t out = shape[0];
    int64_t in = shape[1];
    int64_t ky = shape[2];
    int64_t kx = shape[3];
    int qout = (int)((out+15)/16);
    int64_t rows = in * kx * ky;
    uint32_t * compact;
    if (fromCompact) compact = (uint32_t *)input.data_ptr<int32_t>();
    else {
        compact = new uint32_t[rows * qout];
        if (input.dtype() == torch::kFloat32) kernelTo2BitTernary(input.data_ptr<float>(), compact, out, in, ky, kx);
        else if (input.dtype() == torch::kInt32) kernelTo2BitTernary(input.data_ptr<int32_t>(), compact, out, in, ky, kx);
        else if (input.dtype() == torch::kInt8) kernelTo2BitTernary(input.data_ptr<int8_t>(), compact, out, in, ky, kx);
        else {
            delete [] compact;
            throw std::invalid_argument("Cannot handle the supplied data-type");
        }
    }
    auto eres = entropyEncodeTernary(compact, rows, out);
    if (!fromCompact) delete [] compact;
    if (eres.first == nullptr || eres.second == 0) {
        delete [] eres.first;
        throw std::runtime_error("Cannot encode data");
    }
    auto options = torch::TensorOptions().device(input.device()).dtype(torch::kInt8);
    auto output = torch::empty({eres.second}, options);
    memcpy(output.data_ptr<int8_t>(), eres.first, eres.second);
    delete [] eres.first;
    return output;
}


/**
 * @brief Compactify ternary input weight tensor (either linear or 2D convolution) using simple (2-bit) bit-stuffing
 *
 * @param input Input weight tensor (either for linear or 2D convolution layers)
 *
 * @return 32-bit integer tensor with bit-stuffed values
 *
 * Computes a compact representation for the supplied weight tensor which can be used with custom CUDA kernels.
 * The compact representation performs bit-stuffing of the tensor values (assuming that they are ternary already)
 * and stores 16 values of 2-bit each in a 32-bit value.
 *
 * @see compactTernaryMatrix(), compactTernaryConvKernel()
 */
torch::Tensor compactifyTernary(torch::Tensor input) {
    TORCH_CHECK(input.device().type() == torch::kCPU);
    auto shape = input.sizes();
    TORCH_CHECK(shape.size() <= 2 || shape.size() == 4);
    if (shape.size() <= 2) return compactTernaryMatrix(input);
    if (shape.size() == 4) return compactTernaryConvKernel(input);
    throw std::invalid_argument("Cannot handle the supplied shape");
}


/**
 * @brief Compress ternary input weight tensor (either linear or convolution) using entropy coding
 *
 * @param input Input (integer) tensor data to run entropy-coding on

 * @param fromCompact
 *
 * @return
 *
 *
 */
torch::Tensor compressTernary(torch::Tensor input, std::vector<int64_t> origShape, bool fromCompact) {
    TORCH_CHECK(input.device().type() == torch::kCPU);
    auto shape = input.sizes();
    if (fromCompact) {
        TORCH_CHECK(shape.size() == 2 || shape.size() == 4);
        TORCH_CHECK(input.dtype() == torch::kInt32);
        int64_t rows = (shape.size() == 2) ? shape[0] : shape[0] * shape[1] * shape[2];
        auto options = torch::TensorOptions().device(input.device()).dtype(torch::kInt8);
        if (origShape.size() == 4) {
            int64_t out = origShape[0];
            auto eres = entropyEncodeTernary((const uint32_t *)input.data_ptr<int32_t>(), rows, out);
            if (eres.first == nullptr || eres.second == 0) {
                delete [] eres.first;
                throw std::runtime_error("Cannot encode data");
            }
            auto output = torch::empty({eres.second}, options);
            memcpy(output.data_ptr<int8_t>(), eres.first, eres.second);
            delete [] eres.first;
            return output;
        } else {
            int64_t cols = origShape[0];        // NOTE (mw) matrix is transposed
            auto eres = entropyEncodeTernary((const uint32_t *)input.data_ptr<int32_t>(), rows, cols);
            if (eres.first == nullptr || eres.second == 0) {
                delete [] eres.first;
                throw std::runtime_error("Cannot encode data");
            }
            auto output = torch::empty({eres.second}, options);
            memcpy(output.data_ptr<int8_t>(), eres.first, eres.second);
            delete [] eres.first;
            return output;
        }
    } else {
        TORCH_CHECK(shape.size() <= 2 || shape.size() == 4);
        TORCH_CHECK(input.dtype() == torch::kFloat32);
        if (shape.size() <= 2) return compressTernaryMatrix(input, fromCompact);
        if (shape.size() == 4) return compressTernaryConvKernel(input, fromCompact);
        throw std::invalid_argument("Cannot handle the supplied shape");
    }
}


/**
 * @brief Decompress entropy-coded ternary data into compact (bit-stuffed) representation
 *
 * @param input
 * @param origShape
 *
 * @return
 *
 */
torch::Tensor decompressTernary(torch::Tensor input, std::vector<int64_t> origShape) {
    TORCH_CHECK(input.device().type() == torch::kCPU);
    auto shape = input.sizes();
    TORCH_CHECK(shape.size() == 1);
    TORCH_CHECK(input.dtype() == torch::kInt8);
    TORCH_CHECK(shape[0] >= 8+8+4+6);
    auto options = torch::TensorOptions().device(input.device()).dtype(torch::kInt32);
    int64_t cols = origShape[0];
    int64_t qcols = (cols+15)/16;
    int64_t rows = (origShape.size() == 2) ? origShape[1] : origShape[1] * origShape[2] * origShape[3];
    auto output = (origShape.size() == 4) ? torch::empty({origShape[1], origShape[2], origShape[3], qcols}, options) : torch::empty({rows, qcols}, options);
    uint64_t syms = entropyDecodeTernary((const uint8_t *)input.data_ptr<int8_t>(), (uint32_t *)output.data_ptr<int32_t>(), rows, cols);
    TORCH_CHECK(syms > 0);
    return output;
}


/**
 * @brief Expand compressed/compact ternary weight tensor to 32-bit floating point
 *
 * @param input
 * @param origShape
 * @param entropyDecode
 *
 * @return
 *
 */
torch::Tensor expandTernary(torch::Tensor input, std::vector<int64_t> origShape, bool entropyDecode) {
    auto expand = [&origShape](torch::Tensor & input) -> torch::Tensor {
        auto inshape = input.sizes();
        auto options = torch::TensorOptions().device(input.device()).dtype(torch::kFloat32);
        if (inshape.size() <= 2) {
            auto output = torch::empty(origShape, options);
            compactMatrixTo32BitFloat((uint32_t *)input.data_ptr<int32_t>(), output.data_ptr<float>(), origShape[0], origShape[1]);
            return output;
        }
        if (inshape.size() == 4) {
             auto output = torch::empty(origShape, options);
             compactKernelTo32BitFloat((uint32_t *)input.data_ptr<int32_t>(), output.data_ptr<float>(), origShape[0], origShape[1], origShape[2], origShape[3]);
             return output;
        } else throw std::invalid_argument("Shape needs to be of dim 2 or 4");
    };
    TORCH_CHECK(input.device().type() == torch::kCPU);
    TORCH_CHECK(input.dtype() == torch::kInt32);
    auto inshape = input.sizes();
    if (entropyDecode) {
        TORCH_CHECK(inshape.size() == 1);
        auto tmp = decompressTernary(input, origShape);
        auto shape = tmp.sizes();
        TORCH_CHECK(shape.size() <= 2 || shape.size() == 4);
        return expand(tmp);
    } else {
        TORCH_CHECK(inshape.size() <= 2 || inshape.size() == 4);
        return expand(input);
    }
}



//---------------------------------------- Python Bindings -----------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("compactify_ternary", &compactifyTernary, "Encode tensor with ternary values into compact representation using 16 elements per 32-bit value in the compacted tensor")
       .def("compress_ternary", &compressTernary, "Encode tensor with ternary values into compressed representation using entropy coding")
       .def("decompress_ternary", &decompressTernary, "Decompress tensor with compressed ternary values into compact representation")
       .def("expand_ternary", &expandTernary, "Decode tensor with compact or compressed ternary values into 32-bit float representation");
}

// vim: set expandtab ts=4 sw=4:
