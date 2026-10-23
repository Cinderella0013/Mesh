#pragma once
#include "management_link.h"
#include "receiver_sounding.h"
#include "fresh_sounding.h"
#include "corrected_telemetry.h"
#include "demand_schedule.h"
#include <bit>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace threeway {
// CPU policy runs outside ns-3. Its measured planning latency is explicitly scheduled
// before the command packets; waiting for stdin does not itself advance simulation time.
class PolicyBridge {
public:
    PolicyBridge(const NodeContainer& aps,ManagementEpochs::Readback read,
                 ManagementEpochs::Actuator act,ReceiverSounding& receiver,
                 CorrectedTelemetry& corrected,const std::vector<uint64_t>& nominalBps,
                 double interval,double stop,const std::string& output,uint64_t dropEpoch,int dropNode,
                 const DemandSchedule* demand=nullptr,FreshSounding* fresh=nullptr)
        :m_links(aps,output+".management.csv",aps.Get(0)),m_protocol(m_links,read,act,output+".management-state.csv"),
         m_receiver(receiver),m_corrected(corrected),m_nominal(nominalBps),
         m_period(Seconds(interval)),m_stop(Seconds(stop)),m_trace(output+".policy-command.csv"),
         m_dropEpoch(dropEpoch),m_dropNode(dropNode),m_demand(demand),m_fresh(fresh) {
        NS_ABORT_MSG_IF(nominalBps.size()!=aps.GetN()||!m_trace||dropNode>=int(aps.GetN())||dropNode< -1,
                        "invalid policy bridge configuration");
        m_trace<<"epoch,observedAtNs,availableAtNs,planningLatencyNs,commandSubmitNs,targets\n";
        m_protocol.reportReady=[this](uint64_t seq){ReportReady(seq);};
        m_protocol.ackReady=[this](uint64_t seq){AckReady(seq);};
    }
    void SetActionSettled(std::function<void(uint64_t)> done){m_actionSettled=std::move(done);}
    void Start(){if(!m_fresh)Simulator::Schedule(Seconds(1),&PolicyBridge::Tick,this);}
    void FreshReady(uint64_t seq) {
        NS_ABORT_MSG_IF(!m_fresh||seq!=m_tick+1||m_protocol.Pending()||m_planning,"fresh report sequence or outstanding action");
        Tick();
    }
    void FreshUnavailable(uint64_t seq) {
        NS_ABORT_MSG_IF(!m_fresh||seq!=m_tick+1||m_protocol.Pending()||m_planning,"unavailable acquisition sequence or outstanding action");
        // Advance the absolute opportunity, not the number of successful reports.
        // Tick's previous-window index must still refer to the immediately prior fixed interval.
        ++m_tick;
        std::cout<<"THREEWAY_UNAVAILABLE {\"epoch\":"<<seq<<",\"availableAtNs\":"
                 <<Simulator::Now().GetNanoSeconds()<<",\"reason\":\"radio_not_quiescent\"}"<<std::endl;
    }
private:
    static void Put(std::vector<uint8_t>& p,uint64_t value) {
        for(int b=7;b>=0;--b)p.push_back((value>>(b*8))&255);
    }
    static uint64_t Get(const std::vector<uint8_t>& p,size_t& offset) {
        NS_ABORT_MSG_IF(offset+8>p.size(),"truncated management report payload");uint64_t result=0;
        for(unsigned i=0;i<8;++i)result=(result<<8)|p[offset++];return result;
    }
    static double Double(const std::vector<uint8_t>& p,size_t& offset) {
        double value=std::bit_cast<double>(Get(p,offset));
        NS_ABORT_MSG_IF(!std::isfinite(value),"nonfinite management report feature");return value;
    }
    void Tick() {
        const uint64_t seq=m_tick+1;
        if(m_dropNode>=0)m_links.BlockUp(m_dropNode,false);
        if(!m_protocol.Pending()&&!m_planning) {
            std::vector<std::vector<uint8_t>> payloads;
            for(uint32_t n=0;n<m_nominal.size();++n) {
                std::vector<uint8_t> p;
                if(m_fresh) {
                    Put(p,m_fresh->Epoch());Put(p,m_fresh->OldestNs());Put(p,m_fresh->ReadyNs());
                }
                for(double power:m_fresh?m_fresh->Row(n):m_receiver.Row(n))Put(p,std::bit_cast<uint64_t>(power));
                const uint64_t rate=m_demand?m_demand->At(n,Simulator::Now().GetNanoSeconds()):m_nominal[n];
                Put(p,std::bit_cast<uint64_t>(rate/1e6));
                uint64_t tx=0,rx=0;int64_t a=Seconds(1).GetNanoSeconds(),b=a;
                if(m_tick) {
                    const auto& w=m_corrected.windows.at(n).at(m_tick-1);
                    NS_ABORT_MSG_IF(w.end>Simulator::Now().GetNanoSeconds(),"future policy reward window");
                    tx=w.txBytes;rx=w.rxBytes;a=w.start;b=w.end;
                }
                Put(p,tx);Put(p,rx);
                const double demand=m_demand?m_demand->Bytes(n,a,b):
                    std::max<int64_t>(0,std::min(b,(m_stop-MilliSeconds(100)).GetNanoSeconds())-a)*m_nominal[n]/8e9;
                Put(p,std::bit_cast<uint64_t>(demand));Put(p,a);Put(p,b);payloads.push_back(std::move(p));
            }
            m_protocol.Reports(seq,payloads);
        }
        ++m_tick;
        const auto next=Seconds(1)+m_tick*m_period;
        if(!m_fresh&&next<m_stop-MilliSeconds(100))Simulator::Schedule(next-Simulator::Now(),&PolicyBridge::Tick,this);
    }
    void ReportReady(uint64_t seq) {
        std::ostringstream json;json<<std::setprecision(17);
        uint64_t oldest=0,ready=0;
        json<<"{\"epoch\":"<<seq<<",\"observedAtNs\":"<<m_protocol.ObservedAt()
            <<",\"availableAtNs\":"<<m_protocol.AvailableAt();
        if(m_fresh) {
            size_t at=0;const auto& first=m_protocol.Payloads()[0];
            NS_ABORT_MSG_IF(Get(first,at)!=seq,"fresh management batch mismatch");
            oldest=Get(first,at);ready=Get(first,at);
            NS_ABORT_MSG_IF(ready>uint64_t(m_protocol.ObservedAt())||oldest>=ready,"fresh management timing mismatch");
            json<<",\"acquisition\":{\"batchEpoch\":"<<seq<<",\"oldestAcquisitionNs\":"<<oldest
                <<",\"readyAtNs\":"<<ready<<",\"aggregateEnergy\":{\"valid\":false,\"valueW\":null,\"reason\":\"mixed_channel_window\"}}";
        }
        json<<",\"nodes\":[";
        for(uint32_t n=0;n<m_nominal.size();++n) {
            if(n)json<<',';const auto& p=m_protocol.Payloads()[n];size_t offset=0;
            if(m_fresh) {
                const auto batch=Get(p,offset),start=Get(p,offset),available=Get(p,offset);
                NS_ABORT_MSG_IF(batch!=seq||start!=oldest||available!=ready,"mixed fresh node batches");
            }
            json<<"{\"node\":"<<n<<",\"channel\":"<<unsigned(m_protocol.Channels()[n])<<",\"powersW\":[";
            for(unsigned c=0;c<m_nominal.size()*4;++c){if(c)json<<',';json<<Double(p,offset);}
            json<<"],\"nominalMbps\":"<<Double(p,offset)<<",\"previousTxBytes\":"<<Get(p,offset)
                <<",\"previousDeliveredBytes\":"<<Get(p,offset)<<",\"previousNominalBytes\":"<<Double(p,offset)
                <<",\"previousWindowStartNs\":"<<Get(p,offset)<<",\"previousWindowEndNs\":"<<Get(p,offset)<<'}';
            NS_ABORT_MSG_IF(offset!=p.size(),"unexpected management report payload bytes");
        }
        json<<"]}";std::cout<<"THREEWAY_OBS "<<json.str()<<std::endl;
        std::string line;NS_ABORT_MSG_IF(!std::getline(std::cin,line),"missing policy command for report");
        std::istringstream command(line);uint64_t epoch,latencyNs;std::vector<uint8_t> targets;
        NS_ABORT_MSG_IF(!(command>>epoch>>latencyNs)||epoch!=seq||latencyNs>uint64_t(Seconds(60).GetNanoSeconds()),"invalid policy command identity/planning latency");
        for(uint32_t n=0;n<m_nominal.size();++n) {
            int c;NS_ABORT_MSG_IF(!(command>>c)||c<0||c>=4,"invalid/incomplete policy channel vector");targets.push_back(c);
        }
        std::string extra;NS_ABORT_MSG_IF(command>>extra,"extra policy command fields");
        NS_ABORT_MSG_IF(m_fresh&&uint64_t(Simulator::Now().GetNanoSeconds())+latencyNs-oldest>700000000,
                        "fresh acquisition expired during charged planning");
        if(seq==m_dropEpoch&&m_dropNode>=0)m_links.BlockUp(m_dropNode,true);
        m_trace<<seq<<','<<m_protocol.ObservedAt()<<','<<m_protocol.AvailableAt()<<','<<latencyNs<<','
            <<Simulator::Now().GetNanoSeconds()+latencyNs<<',';
        for(uint32_t n=0;n<targets.size();++n){if(n)m_trace<<';';m_trace<<unsigned(targets[n]);}
        m_trace<<'\n';m_trace.flush();m_planning=true;
        Simulator::Schedule(NanoSeconds(latencyNs),[this,seq,targets]{
            m_planning=false;NS_ABORT_MSG_IF(!m_protocol.Submit(seq,targets),"policy vector submission failed");
        });
    }
    void AckReady(uint64_t seq) {
        std::ostringstream json;
        json<<"{\"epoch\":"<<seq<<",\"availableAtNs\":"<<m_protocol.AvailableAt()<<",\"channels\":[";
        for(uint32_t n=0;n<m_nominal.size();++n){if(n)json<<',';json<<unsigned(m_protocol.Channels()[n]);}
        json<<"],\"statuses\":[";
        for(uint32_t n=0;n<m_nominal.size();++n){if(n)json<<',';json<<unsigned(m_protocol.Statuses()[n]);}
        json<<"]}";std::cout<<"THREEWAY_ACK "<<json.str()<<std::endl;
        if(m_actionSettled)m_actionSettled(seq);
    }
    std::function<void(uint64_t)> m_actionSettled;
    ManagementLinks m_links;ManagementEpochs m_protocol;ReceiverSounding& m_receiver;
    CorrectedTelemetry& m_corrected;std::vector<uint64_t> m_nominal;
    Time m_period,m_stop;std::ofstream m_trace;uint64_t m_tick=0,m_dropEpoch;int m_dropNode;
    bool m_planning=false;
    const DemandSchedule* m_demand=nullptr;
    FreshSounding* m_fresh=nullptr;
};
} // namespace threeway
