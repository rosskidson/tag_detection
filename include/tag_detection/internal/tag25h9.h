#pragma once

#include <vector>

namespace tag_detection {

// Tag bits are read like a book; left to right, top to bottom. Most significant bit is top left.

const std::vector<unsigned long long> t25h9{
    0x155cbf1LL, 0x1e4d1b6LL, 0x17b0b68LL, 0x1eac9cdLL, 0x12e14ceLL, 0x3548bbLL,  0x7757e6LL,
    0x1065dabLL, 0x1baa2e7LL, 0xdea688LL,  0x81d927LL,  0x51b241LL,  0xdbc8aeLL,  0x1e50e19LL,
    0x15819d2LL, 0x16d8282LL, 0x163e035LL, 0x9d9b81LL,  0x173eec4LL, 0xae3a09LL,  0x5f7c51LL,
    0x1a137fcLL, 0xdc9562LL,  0x1802e45LL, 0x1c3542cLL, 0x870fa4LL,  0x914709LL,  0x16684f0LL,
    0xc8f2a5LL,  0x833ebbLL,  0x59717fLL,  0x13cd050LL, 0xfa0ad1LL,  0x1b763b0LL, 0xb991ceLL};

}  // namespace tag_detection
