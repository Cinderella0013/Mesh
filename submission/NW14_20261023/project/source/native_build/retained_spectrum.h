#pragma once
#include "ns3/wifi-module.h"
#include "ns3/spectrum-module.h"

// Keep simulator interference history per band; there is still one active Wi-Fi PHY.
inline void AddRetainedSpectrum(ns3::SpectrumWifiPhyHelper& helper,ns3::Ptr<ns3::SpectrumChannel> channel) {
    for(unsigned c=0;c<4;++c)
        helper.AddChannel(channel,ns3::FrequencyRange{ns3::MHz_u{5170.+20*c},ns3::MHz_u{5190.+20*c}});
    helper.Set("TrackSignalsFromInactiveInterfaces",ns3::BooleanValue(true));
}
inline void ConfigureRetainedSpectrum(ns3::Ptr<ns3::SpectrumWifiPhy> phy) {
    NS_ABORT_MSG_IF(phy->GetDevice()->GetNPhys()!=1||phy->GetChannelWidth()!=ns3::MHz_u{20},"retention qualified for single 20MHz PHY");
    for(unsigned c=0;c<4;++c)
        if(phy->GetChannelNumber()!=36+4*c)
            phy->ConfigureInterface({ns3::MHz_u{5180.+20*c}},ns3::MHz_u{20});
}
inline bool RetainedSpectrumReady(ns3::Ptr<ns3::WifiPhy> base) {
    auto phy=ns3::DynamicCast<ns3::SpectrumWifiPhy>(base);
    if(!phy||phy->GetSpectrumPhyInterfaces().size()!=4)return false;
    ns3::BooleanValue tracking;phy->GetAttribute("TrackSignalsFromInactiveInterfaces",tracking);
    if(!tracking.Get()||phy->GetDevice()->GetNPhys()!=1)return false;
    for(unsigned c=0;c<4;++c) {
        auto range=ns3::FrequencyRange{ns3::MHz_u{5170.+20*c},ns3::MHz_u{5190.+20*c}};
        auto it=phy->GetSpectrumPhyInterfaces().find(range);
        if(it==phy->GetSpectrumPhyInterfaces().end()||it->second->GetCenterFrequencies()!=std::vector<ns3::MHz_u>{ns3::MHz_u{5180.+20*c}}||it->second->GetChannelWidth()!=ns3::MHz_u{20})return false;
    }
    return true;
}
