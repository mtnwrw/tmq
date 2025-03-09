//--------------------------------------------------------------------------------------------------
// TMQ                                                                          (c) TMQ Authors 2025
//--------------------------------------------------------------------------------------------------
// Compressed Bitstreams (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MPL-2.0
//--------------------------------------------------------------------------------------------------

#pragma once

//-------------------------------------- Project  Headers ------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <cstdint>
#include <vector>
#include <cstdlib>
#include <type_traits>

//------------------------------------- Public Declarations ----------------------------------------


/**
 * @brief Bitstream class for output
 *
 * This class provides functionality to create and maintain a stream of bits for serialization into
 * an I/O device. It performs all the nitty-gritty bit-stuffing and provides a neat and simple
 * interface.
 */
class WriteStream {
 public:
    WriteStream() {
        buffer_.reserve(32768);
        buffer_.emplace_back(0);
    }

    void appendRaw(const uint8_t *ptr, int numBytes) {
        for (int i=0; i<numBytes; i++) bitAppender(*ptr++, 8);
    }

    void appendRaw(const std::vector<uint8_t> & data) {
        for (size_t i=0; i < data.size(); i++) bitAppender(data.at(i), 8);
    }

    void appendByte(uint8_t byte) {
        bitAppender(byte, 8);
    }

    void appendBits(uint32_t data, int bits) {
        assert(bits > 0);
        while (bits > 0)
        {
            int shift = (bits >= 8) ? bits-8 : 0;
            bitAppender((data >> shift) & 0xFF, (bits <= 8) ? bits : 8);
            bits -= 8;
        }
    }

    [[nodiscard]] size_t numBits() const {
        return bits_;
    }

    [[nodiscard]] size_t numBytes() const {
        return buffer_.size();
    }

    [[nodiscard]] const std::vector<uint8_t> & buffer() const {
        return buffer_;
    }

 private:

    void bitAppender(uint8_t data, int bits) {
        assert(bits <= 8);
        bits_ += bits;
        if (bitOffset_ + bits > 8) {
            int rshift = bitOffset_ + bits - 8;
            uint8_t mix = buffer_[offset_] | (data>>rshift);
            buffer_[offset_++] = mix;
            bits -= 8-bitOffset_;
            buffer_.emplace_back(data << (8-bits));
            bitOffset_ = bits;
        } else {
            int shift = 8-bits-bitOffset_;
            assert(shift >= 0);
            assert(shift < 8);
            uint8_t mix = buffer_[offset_] | (data<<shift);
            buffer_[offset_] = mix;
            bitOffset_ += bits;
            if (bitOffset_ == 8) {
                offset_++;
                bitOffset_ = 0;
                buffer_.emplace_back(0);
            }
        }
    }

    std::vector<uint8_t> buffer_;
    size_t bits_ = 0;
    size_t offset_ = 0;
    int bitOffset_ = 0;
};



/**
 * @brief Bitstream class for input
 *
 * This class provides functionality to create and maintain an input bitstream from an I/O device.
 * It performs all the nitty-gritty bit-stuffing and provides a neat and simple interface.
 */
template<class BTYPE>
class ReadStream {
    using BufferType = typename std::conditional<
        std::is_pointer<BTYPE>::value,
        typename std::add_pointer<typename std::add_const<typename std::remove_pointer<BTYPE>::type>::type>::type,
        const BTYPE
    >::type;
 public:

    template<typename T = BTYPE, typename std::enable_if<std::is_pointer<T>::value>::type* = nullptr>
    ReadStream():buffer_(nullptr), bits_(0) {
        static_assert(std::is_pointer<BTYPE>::value);
    }

    template<typename T = BTYPE, typename std::enable_if<!std::is_pointer<T>::value>::type* = nullptr>
    ReadStream(): bits_(0) {
    }

    ReadStream(const BTYPE & buffer, size_t bits) : buffer_(buffer), bits_(bits) {
        bytes_ = (bits+7)/8;
    }

    [[nodiscard]] uint8_t getByte() const {
        if (offset_ >= bytes_) return 0;
        if (bitOffset_ == 0) {
            return buffer_[offset_++];
        } else {
            int left = buffer_[offset_++];
            int right = (offset_ < bytes_) ? buffer_[offset_] : 0;
            return (left << bitOffset_) | (right >> (8-bitOffset_));
        }
    }

    [[nodiscard]] uint32_t getBits(int bits) const {
        assert(bits <= 32);
        assert(bits > 0);
        uint32_t shiftreg = 0;
        if (bitOffset_ != 0) {
            shiftreg = buffer_[offset_] & ((1 << (8 - bitOffset_)) - 1);
            if (bits < 8 - bitOffset_) {
                bitOffset_ += bits;
                return shiftreg >> (8 - bitOffset_ - bits);
            }
            bits -= (8 - bitOffset_);
            offset_++;
            bitOffset_ = 0;
        }
        while (bits >= 8) {
            shiftreg = (shiftreg << 8) | buffer_[offset_++];
            bits -= 8;
        }
        if (bits > 0) {
            shiftreg = ((shiftreg << 8) | buffer_[offset_]) >> (8-bits);
            bitOffset_ = bits;
        }
        return shiftreg;
    }

 private:
    const BufferType buffer_;
    size_t bits_ = 0;
    size_t bytes_ = 0;
    mutable size_t offset_ = 0;
    mutable int bitOffset_ = 0;
};

// vim: set expandtab ts=4 sw=4:
