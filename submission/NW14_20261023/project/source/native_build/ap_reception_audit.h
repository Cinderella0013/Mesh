#pragma once
#include "ns3/wifi-module.h"
#include <cmath>
#include <fstream>
#include <iomanip>
using namespace ns3;

// Offline simulator-known headers identify intended links even for failed frames.
// No values from this recorder are passed to a runtime controller.
class ApReceptionAudit {
public:
    struct Counts { uint64_t ok=0,failed=0,drops=0,ignored=0,invalid=0; };
    std::vector<Counts> counts;
    std::vector<Mac48Address> aps,stas;
    std::ofstream events;
    std::string path;
    int64_t stop;
    uint64_t rows=0;
    ApReceptionAudit(uint32_t n,double until,std::string output)
        :counts(n),aps(n),stas(n),events(output),path(output),stop(Seconds(until).GetNanoSeconds()) {
        NS_ABORT_MSG_IF(!events,"cannot open AP reception audit");
        events<<std::setprecision(17);
        events<<"node,timeNs,event,packetUid,ppduUid,success,reasonCode,snrLinear,rssiDbm,signalValid,mode\n";
    }
    void Pair(uint32_t n,Mac48Address ap,Mac48Address sta) {aps.at(n)=ap;stas.at(n)=sta;}
    bool InWindow() const {auto now=Simulator::Now().GetNanoSeconds();return now>=1000000000&&now<stop;}
    bool Intended(uint32_t n,const WifiMacHeader& h) {
        if(h.IsData()&&!h.GetAddr1().IsGroup()&&h.GetAddr1()==aps.at(n)&&h.GetAddr2()==stas.at(n))return true;
        ++counts.at(n).ignored;return false;
    }
    void Prefix(uint32_t n,const char* event,uint64_t uid) {
        // ponytail: bounded diagnostic event log; review scope before raising cap.
        NS_ABORT_MSG_IF(++rows>200000,"AP audit event cap exceeded; no silent truncation");
        events<<n<<','<<Simulator::Now().GetNanoSeconds()<<','<<event<<','<<uid<<',';
    }
    static void Outcome(ApReceptionAudit* self,uint32_t n,Ptr<const WifiPpdu> ppdu,
                        RxSignalInfo info,const WifiTxVector& tx,const std::vector<bool>& outcomes) {
        if(!self->InWindow())return;
        const auto psdu=ppdu->GetPsdu();
        NS_ABORT_MSG_IF(!psdu||psdu->GetNMpdus()!=outcomes.size(),"AP outcome cardinality mismatch");
        size_t i=0;
        for(const auto& mpdu:*psdu) {
            const bool success=outcomes.at(i++);
            if(!self->Intended(n,mpdu->GetHeader()))continue;
            auto& c=self->counts.at(n);success?++c.ok:++c.failed;
            // This runtime initializes RxSignalInfo only on payload success.
            const double rssi=success?static_cast<double>(info.rssi):0;
            const bool valid=success&&std::isfinite(info.snr)&&info.snr>0&&std::isfinite(rssi);
            c.invalid+=!valid;
            self->Prefix(n,"rx_outcome",mpdu->GetPacket()->GetUid());
            self->events<<ppdu->GetUid()<<','<<success<<",,";
            if(valid)self->events<<info.snr;
            self->events<<',';if(valid)self->events<<rssi;
            self->events<<','<<valid<<','<<tx.GetMode().GetUniqueName()<<'\n';
        }
    }
    static void Drop(ApReceptionAudit* self,uint32_t n,Ptr<const Packet> packet,WifiPhyRxfailureReason reason) {
        if(!self->InWindow())return;
        WifiMacHeader header;packet->PeekHeader(header);
        if(!self->Intended(n,header))return;
        ++self->counts.at(n).drops;
        self->Prefix(n,"phy_drop",packet->GetUid());
        // No PPDU UID, SNR/RSSI or PHY mode supplied by this trace. Do not invent them.
        self->events<<",,"<<static_cast<int>(reason)<<",,,0,\n";
    }
    void Finish() {
        events.flush();NS_ABORT_MSG_IF(!events,"cannot write AP reception events");
        std::ofstream summary(path+".summary.csv");
        summary<<"node,rxOkMpdus,rxFailedMpdus,phyDropEvents,ignoredEvents,invalidSignalOutcomes\n";
        for(uint32_t n=0;n<counts.size();++n) {
            const auto& c=counts[n];summary<<n<<','<<c.ok<<','<<c.failed<<','<<c.drops<<','<<c.ignored<<','<<c.invalid<<'\n';
        }
        summary.flush();NS_ABORT_MSG_IF(!summary,"cannot write AP reception summary");
    }
};
