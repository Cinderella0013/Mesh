#pragma once
#include "ns3/wifi-module.h"
#include <fstream>
#include <map>
#include <vector>
using namespace ns3;

// Evaluator-only application UID accounting. No packet record enters a policy.
// UID survives packet copies/encapsulation in this unfragmented UDP benchmark;
// native qualification must reconcile every receive against its original send.
class PacketCohort {
public:
    struct Record {
        int64_t sent=-1,received=-1,drop=-1,ack=-1;
        uint32_t bytes=0;
        uint64_t duplicates=0,acks=0,drops=0;
        int reason=-1;
    };
    int64_t start=Seconds(1).GetNanoSeconds(),stop;
    std::vector<std::map<uint64_t,Record>> records;
    std::string output;
    PacketCohort(uint32_t n,double until,std::string path)
        :stop(Seconds(until).GetNanoSeconds()),records(n),output(path) {
        NS_ABORT_MSG_IF(!n||stop<=start,"invalid application cohort horizon");
    }
    bool InWindow() const {auto now=Simulator::Now().GetNanoSeconds();return start<=now&&now<stop;}
    void Tx(uint32_t n,Ptr<const Packet> packet) {
        if(!InWindow())return;
        auto& r=records.at(n)[packet->GetUid()];
        NS_ABORT_MSG_IF(r.sent>=0,"application UID sent twice");
        r.sent=Simulator::Now().GetNanoSeconds();r.bytes=packet->GetSize();
    }
    void Rx(uint32_t n,Ptr<const Packet> packet) {
        if(!InWindow())return;
        auto found=records.at(n).find(packet->GetUid());
        NS_ABORT_MSG_IF(found==records[n].end()||found->second.sent<0,"unmatched application receive UID");
        auto& r=found->second;
        NS_ABORT_MSG_IF(r.bytes!=packet->GetSize(),"application payload changed during transit");
        if(r.received>=0){++r.duplicates;return;}
        r.received=Simulator::Now().GetNanoSeconds();
        NS_ABORT_MSG_IF(r.received<r.sent,"application receive precedes send");
    }
    void Mac(uint32_t n,Ptr<const WifiMpdu> packet,bool dropped,int reason=-1) {
        if(!InWindow())return;
        const auto& h=packet->GetHeader();
        if(!h.IsData()||h.GetAddr1().IsGroup())return;
        auto& r=records.at(n)[packet->GetPacket()->GetUid()];
        const auto now=Simulator::Now().GetNanoSeconds();
        if(dropped){++r.drops;if(r.drop<0){r.drop=now;r.reason=reason;}}
        else {++r.acks;if(r.ack<0)r.ack=now;}
    }
    void Write() const {
        NS_ABORT_MSG_IF(Simulator::Now().GetNanoSeconds()!=stop,"application cohort must close at declared stop");
        std::ofstream o(output),summary(output+".summary.csv");
        NS_ABORT_MSG_IF(!o||!summary,"cannot open application cohort outputs");
        o<<"node,packetUid,bytes,sentAtNs,receivedAtNs,delayNs,status,censorAtNs,censorAgeNs,macDropAtNs,macDropReason,macDropEvents,macAckAtNs,macAckEvents,duplicateReceives\n";
        summary<<"node,applicationTxPackets,applicationDeliveredPackets,undeliveredMacTerminalPackets,undeliveredCensoredPackets,duplicateReceives,nonApplicationMacUids\n";
        for(uint32_t n=0;n<records.size();++n) {
            uint64_t tx=0,rx=0,terminal=0,censored=0,duplicates=0,other=0;
            for(const auto& [uid,r]:records[n]) {
                if(r.sent<0){++other;continue;}
                ++tx;duplicates+=r.duplicates;
                const bool received=r.received>=0;
                const char* status=received?"delivered":r.drop>=0?"undelivered_mac_terminal":"undelivered_censored";
                received?++rx:r.drop>=0?++terminal:++censored;
                o<<n<<','<<uid<<','<<r.bytes<<','<<r.sent<<',';
                if(received)o<<r.received;
                o<<',';if(received)o<<r.received-r.sent;
                o<<','<<status<<',';if(!received)o<<stop;
                o<<',';if(!received)o<<stop-r.sent;
                o<<',';if(r.drop>=0)o<<r.drop;
                o<<',';if(r.drop>=0)o<<r.reason;
                o<<','<<r.drops<<',';if(r.ack>=0)o<<r.ack;
                o<<','<<r.acks<<','<<r.duplicates<<'\n';
            }
            summary<<n<<','<<tx<<','<<rx<<','<<terminal<<','<<censored<<','<<duplicates<<','<<other<<'\n';
        }
        o.flush();summary.flush();NS_ABORT_MSG_IF(!o||!summary,"cannot write application cohort");
    }
    static void Ack(PacketCohort* self,uint32_t n,Ptr<const WifiMpdu> p){self->Mac(n,p,false);}
    static void Drop(PacketCohort* self,uint32_t n,WifiMacDropReason r,Ptr<const WifiMpdu> p){self->Mac(n,p,true,int(r));}
};
