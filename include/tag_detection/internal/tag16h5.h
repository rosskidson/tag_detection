#pragma once

#include <vector>

namespace tag_detection {

// Tag bits are read like a book; left to right, top to bottom. Most significant bit is top left.

const std::vector<uint64_t> t16h5{0x231bLL, 0x2ea5LL, 0x346aLL, 0x45b9LL, 0x79a6LL, 0x7f6bLL,
                                  0xb358LL, 0xe745LL, 0xfe59LL, 0x156dLL, 0x380bLL, 0xf0abLL,
                                  0x0d84LL, 0x4736LL, 0x8c72LL, 0xaf10LL, 0x093cLL, 0x93b4LL,
                                  0xa503LL, 0x468fLL, 0xe137LL, 0x5795LL, 0xdf42LL, 0x1c1dLL,
                                  0xe9dcLL, 0x73adLL, 0xad5fLL, 0xd530LL, 0x07caLL, 0xaf2eLL};

}  // namespace tag_detection
