//--------------------------------------------------------------------------------------------------
// TMQ                                                                          (c) TMQ Authors 2025
//--------------------------------------------------------------------------------------------------
// Entropy Coder (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MPL-2.0
//--------------------------------------------------------------------------------------------------

#pragma once

//-------------------------------------- Project  Headers ------------------------------------------

#include "bitstream.h"

//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <cstdint>
#include <vector>
#include <cstdlib>

//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Minimal range encoder implementation
 *
 * This class constitutes a minimal implementation of a multi-symbol range coder which uses a 16-bit
 * PMF as symbol probabilities. It outputs the entropy-coded data into a WriteStream object which
 * can be queried from this object. Operation is simple:
 *
 * @code
 *  RangeCoder<uint8_t> coder;
 *  for (int i=0; i < numSymbols; i++) coder.encoder(symbol);
 *  coder.flush();   // indicate end-of-stream
 * @endcode
 *
 * @note This coder and its implementation should be unencumbered by patents.
 *
 * @see WriteStream, RangeDecoder
 */
template<typename T>
class RangeEncoder {
    struct state {
        uint32_t low = 0;
        uint32_t range = 0xFFFFFFFF;
    };
 public:
    /**
     * @brief Constructor
     *
     * @param pmf Probability mass function for the symbols, where data is normalized to 16-bit unsigned integer
     *            values, i.e. probability of 1 corresponds to 65535.
     *
     * @warning The supplied \p pmf \e must have a probability value for each symbol, failure to provide a value
     *            will result in crashes and non-deterministic behaviour.
     */
    explicit RangeEncoder(const std::vector<uint16_t> & pmf) : pmf_(pmf) {
        cmf_.resize(pmf.size());
        for (size_t i=1; i < pmf.size(); i++) cmf_[i] = cmf_[i-1] + pmf[i-1];
        histoTotal_ = cmf_.back() + pmf_.back();
    }

    /**
     * @brief Encode symbol to output bitstream
     *
     * @param symbol Symbol to encode
     */
    void encode(T symbol) {
        assert(pmf_[symbol] > 0);
        assert(histoTotal_ <= 0x10000);
        state_.range /= histoTotal_;
        assert(state_.range > 0);
        state_.low += cmf_[symbol] * state_.range;
        state_.range *= pmf_[symbol];
        assert(pmf_[symbol] > 0);
        assert(state_.range > 0);
        while ((state_.low ^ (state_.low + state_.range)) < 0x01000000 ||
               (state_.range < 0x00010000 && ((state_.range= -state_.low & 0xFFFF),1))) {
            stream_.appendByte(state_.low >> 24);
            state_.range <<= 8;
            state_.low <<= 8;
            assert(state_.range > 0);
        }
    }

    /**
     * @brief Flush output bitstream
     *
     * @note Must be called when done with symbol encoding to flush out the remaining bits in the normalization buffer
     */
    void flush() {
        for (int i=0; i<4; i++) {
            stream_.appendByte(state_.low >> 24);
            state_.low <<= 8;
        }
        state_.range = 0xFFFFFFFF;
    }

    [[nodiscard]] const WriteStream& stream() const {
        return stream_;
    }

 private:
    state state_;                   //!< Encoder state
    WriteStream stream_;            //!< Output bytestream
    uint32_t histoTotal_ = 0;       //!< Total number of counts in the PMF
    std::vector<uint16_t> pmf_;     //!< Probability mass function for symbols, not particularly normalized so it is more like a histogram, but entries have to fit in 16 bits
    std::vector<uint32_t> cmf_;     //!< Cumulative histogram of symbols (except for last one), also not particularly normalized
};




/**
 * @brief Minimal range decoder implementation
 *
 * This class implements a simple range decoder which runs bytewise renormalization on the input streams.
 *
 * @see ReadStream, RangeEncoder
 */
template<typename T, class B>
class RangeDecoder {
    struct state {
        uint32_t low = 0;
        uint32_t range = 0xFFFFFFFF;
        uint32_t shiftReg = 0;
    };
 public:
    /**
     * @brief Constructor
     *
     * @param pmf Probability mass function for the symbols, where data is normalized to 16-bit unsigned integer
     *            values, i.e. probability of 1 corresponds to 65535. Data must have been decoded from stream already
     *            or hardcoded in the source.
     *
     * @param stream Input data stream to read compressed symbols from
     *
     * @see ReadStream
     */
    RangeDecoder(const std::vector<uint16_t> & pmf, const B & stream) : pmf_(pmf), stream_(stream) {
        cmf_.resize(pmf.size());
        for (size_t i=1; i < pmf.size(); i++) cmf_[i] = cmf_[i-1] + pmf[i-1];
        histoTotal_ = cmf_.back() + pmf_.back();
        state_.low = 0;
        state_.shiftReg = 0;
        for (int i=0; i<4; i++) state_.shiftReg = (state_.shiftReg << 8) | stream_.getByte();
        state_.range = 0xFFFFFFFF;
    }

    /**
     * @brief Decode single symbol from input bitstream
     *
     * @return Symbol decoded from stream
     *
     * @see findSymbol()
     */
    T decode() {
        state_.range /= histoTotal_;
        uint32_t freq = (state_.shiftReg - state_.low) / (state_.range);
        T sym = findSymbol(freq, 0, (int)cmf_.size());
        state_.low += cmf_[sym] * state_.range;
        state_.range *= pmf_[sym];
        // renormalize
        while ((state_.low ^ (state_.low + state_.range)) < 0x01000000 ||
               (state_.range < 0x00010000 && ((state_.range = -state_.low & 0xFFFF), 1))) {
            state_.shiftReg = (state_.shiftReg<<8) | stream_.getByte();
            state_.range <<= 8;
            state_.low <<= 8;
        }
        return sym;
    }

 private:
    /**
     * @brief Find corresponding symbol in cumulative probability function by binary search
     */
    T findSymbol(uint32_t freq, int low, int high) const {
        // TODO (mw) use a non-recursive version instead
        int mid = (low + high)/2;
        if (cmf_[mid] > freq) {
            return findSymbol(freq, low, mid);
        } else {
            if (mid >= (int)cmf_.size()-1 || cmf_[mid+1] > freq) return (T)mid;
            else return findSymbol(freq, mid, high);
        }
    }

    state state_;                   //!< Decoder state
    uint32_t histoTotal_ = 0;       //!< Total number of counts in the PMF
    std::vector<uint16_t> pmf_;     //!< Probability mass function for symbols, not particularly normalized so it is more like a histogram, but entries have to fit in 16 bits
    std::vector<uint32_t> cmf_;     //!< Cumulative histogram of symbols (except for last one), also not particularly normalized
    const B stream_;                //!< Input bytestream
};

// vim: set expandtab ts=4 sw=4:
