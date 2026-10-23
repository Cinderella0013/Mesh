#pragma once
#include "retained_spectrum.h"

// Observe the real virtual registration calls; keep the original radio implementation.
class AcquisitionSpectrumChannel : public ns3::MultiModelSpectrumChannel {
public:
    static ns3::TypeId GetTypeId() {
        static ns3::TypeId tid=ns3::TypeId("ns3::AcquisitionSpectrumChannel")
            .SetParent<ns3::MultiModelSpectrumChannel>().AddConstructor<AcquisitionSpectrumChannel>();
        return tid;
    }
    void AddRx(ns3::Ptr<ns3::SpectrumPhy> phy) override {
        const auto key=ns3::PeekPointer(phy);
        const auto uid=phy->GetRxSpectrumModel()->GetUid();
        const auto previous=m_receivers.find(key);
        const auto since=previous!=m_receivers.end()&&previous->second.model==uid
            ?previous->second.since:ns3::Simulator::Now();
        ns3::MultiModelSpectrumChannel::AddRx(phy);
        m_receivers[key]={uid,since};
        NS_ABORT_MSG_IF(m_receivers.size()!=GetNDevices(),"receiver bookkeeping mismatch");
    }
    void RemoveRx(ns3::Ptr<ns3::SpectrumPhy> phy) override {
        ns3::MultiModelSpectrumChannel::RemoveRx(phy);
        m_receivers.erase(ns3::PeekPointer(phy));
        NS_ABORT_MSG_IF(m_receivers.size()!=GetNDevices(),"receiver removal mismatch");
    }
    bool HasHistory(ns3::Ptr<ns3::SpectrumPhy> phy,ns3::Time coverage) const {
        auto it=m_receivers.find(ns3::PeekPointer(phy));
        return it!=m_receivers.end()&&it->second.model==phy->GetRxSpectrumModel()->GetUid()
            &&ns3::Simulator::Now()-it->second.since>=coverage;
    }
protected:
    void DoDispose() override {m_receivers.clear();ns3::MultiModelSpectrumChannel::DoDispose();}
private:
    struct Registration {ns3::SpectrumModelUid_t model;ns3::Time since;};
    std::map<const ns3::SpectrumPhy*,Registration> m_receivers;
};

inline bool TrackingWindowReady(ns3::Ptr<ns3::WifiPhy> phy) {
    if(!RetainedSpectrumReady(phy))return false;
    auto spectrum=ns3::DynamicCast<ns3::SpectrumWifiPhy>(phy);
    for(auto& [range,iface]:spectrum->GetSpectrumPhyInterfaces()) {
        auto channel=ns3::DynamicCast<AcquisitionSpectrumChannel>(iface->GetChannel());
        if(!channel||!channel->HasHistory(iface,ns3::NanoSeconds(100000708)))return false;
    }
    return true;
}
