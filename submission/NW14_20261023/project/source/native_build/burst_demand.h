#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>

// The fixed 1..9.9 s horizon contains exactly 35 complete 50 ms ON slots.
inline uint64_t BurstPeakBps(uint64_t nominal) {
    if(!nominal || nominal>1000000000)throw std::invalid_argument("burst nominal outside domain");
    return nominal*178/35;
}
inline double BurstNominalBytes(uint64_t nominal,int64_t a,int64_t b) {
    if(a>b)throw std::invalid_argument("reversed nominal interval");
    const int64_t elapsed=std::max<int64_t>(0,std::min<int64_t>(b,9900000000)-std::max<int64_t>(a,1000000000));
    return uint64_t(elapsed)*nominal/8e9;
}
