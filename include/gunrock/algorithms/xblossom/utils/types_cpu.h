// Vendored from X-Blossom (src/include/utils/types_cpu.h) for the Gunrock X-Blossom app.
#ifndef TYPES_CPU_H
#define TYPES_CPU_H

#include <cstdint>
#include <utility>
#include <vector>

using index_t = int;
// Offsets into the path buffer, the cycle buffer and the blossom lists: 64-bit,
// so a cycle buffer may exceed 2^31 entries (int32 offsets could not run Orkut).
using offset_t = long long;
using byte_t = uint8_t;

template<typename T>
using HVector = std::vector<T>;

template<typename T1, typename T2>
using HPair = std::pair<T1, T2>;

#endif //TYPES_CPU_H
