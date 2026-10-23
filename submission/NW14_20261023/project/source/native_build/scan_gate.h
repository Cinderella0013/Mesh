#pragma once
#include "ns3/wifi-phy-state.h"
inline bool ScanRetunable(ns3::WifiPhyState state) {
    return state==ns3::WifiPhyState::IDLE || state==ns3::WifiPhyState::CCA_BUSY;
}
