#pragma once
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/simple-channel.h"
#include "ns3/simple-net-device.h"
#include <algorithm>
#include <functional>
#include <fstream>
#include <limits>
#include <vector>

namespace threeway {
using namespace ns3;

class ManagementHeader : public Header {
public:
    enum Kind {REPORT=1,COMMAND=2,ACK=3};
    enum Status {OK=0,STALE=1,REJECTED=2,BUSY=3,ILLEGAL=4};
    uint8_t kind=0,status=0,channel=0,reserved=0;
    uint32_t node=0,magic=0x33574131;
    uint64_t seq=0,stampNs=0;
    static TypeId GetTypeId() {
        static TypeId tid=TypeId("threeway::ManagementHeader").SetParent<Header>()
            .AddConstructor<ManagementHeader>();return tid;
    }
    TypeId GetInstanceTypeId() const override{return GetTypeId();}
    uint32_t GetSerializedSize() const override{return 28;}
    void Serialize(Buffer::Iterator i) const override {
        i.WriteHtonU32(magic);i.WriteU8(kind);i.WriteU8(status);i.WriteU8(channel);i.WriteU8(reserved);
        i.WriteHtonU32(node);i.WriteHtonU64(seq);i.WriteHtonU64(stampNs);
    }
    uint32_t Deserialize(Buffer::Iterator i) override {
        magic=i.ReadNtohU32();kind=i.ReadU8();status=i.ReadU8();channel=i.ReadU8();reserved=i.ReadU8();
        node=i.ReadNtohU32();seq=i.ReadNtohU64();stampNs=i.ReadNtohU64();return 28;
    }
    void Print(std::ostream& o) const override{o<<unsigned(kind)<<':'<<node<<':'<<seq;}
};

class ManagementChannel : public SimpleChannel {
public:
    std::function<void(Ptr<Packet>,Ptr<SimpleNetDevice>)> transmitted;
    void Send(Ptr<Packet> p,uint16_t protocol,Mac48Address to,Mac48Address from,
              Ptr<SimpleNetDevice> sender) override {
        transmitted(p,sender);SimpleChannel::Send(p,protocol,to,from,sender);
    }
};

class ManagementLinks {
public:
    using Receiver=std::function<void(const ManagementHeader&,const std::vector<uint8_t>&)>;
    Receiver coordinator,endpoint;
    ManagementLinks(const NodeContainer& aps,const std::string& output,Ptr<Node> coordinatorNode=nullptr):m_trace(output) {
        NS_ABORT_MSG_IF(!aps.GetN()||!m_trace,"invalid management links/output");
        m_trace<<"timeNs,event,node,direction,kind,seq,channel,status,stampNs,bytes,packetUid,payloadHash\n";
        auto central=coordinatorNode?coordinatorNode:CreateObject<Node>();
        for(uint32_t n=0;n<aps.GetN();++n) {
            auto channel=CreateObject<ManagementChannel>();
            channel->SetAttribute("Delay",TimeValue(MilliSeconds(1)));
            auto cn=Device(central,channel),ap=Device(aps.Get(n),channel);
            cn->SetReceiveCallback(MakeBoundCallback(&ManagementLinks::ReceiveThunk,this,n,true));
            ap->SetReceiveCallback(MakeBoundCallback(&ManagementLinks::ReceiveThunk,this,n,false));
            channel->transmitted=[this,n,ap](Ptr<Packet> p,Ptr<SimpleNetDevice> sender){
                Trace("tx",n,sender==ap,p);
            };
            m_cn.push_back(cn);m_ap.push_back(ap);m_channels.push_back(channel);
        }
    }
    uint32_t Size() const{return m_ap.size();}
    bool Send(const ManagementHeader& h,const std::vector<uint8_t>& payload={}) {
        NS_ABORT_MSG_IF(h.node>=Size()||!Valid(h)||payload.size()+28>1500,"invalid outgoing management message");
        const bool up=h.kind!=ManagementHeader::COMMAND;
        auto p=Create<Packet>(payload.data(),payload.size());p->AddHeader(h);
        auto source=up?m_ap[h.node]:m_cn[h.node],target=up?m_cn[h.node]:m_ap[h.node];
        const bool accepted=source->Send(p,target->GetAddress(),0x88b5);
        Trace(accepted?"enqueue":"enqueue_failed",h.node,up,p);return accepted;
    }
    void BlockUp(uint32_t node,bool blocked) {
        NS_ABORT_MSG_IF(node>=Size(),"invalid blocked management node");
        if(blocked)m_channels[node]->BlackList(m_ap[node],m_cn[node]);
        else m_channels[node]->UnBlackList(m_ap[node],m_cn[node]);
    }
private:
    static Ptr<SimpleNetDevice> Device(Ptr<Node> node,Ptr<SimpleChannel> channel) {
        auto d=CreateObject<SimpleNetDevice>();d->SetAddress(Mac48Address::Allocate());
        d->SetAttribute("DataRate",DataRateValue(DataRate("10Mbps")));
        d->SetAttribute("PointToPointMode",BooleanValue(true));d->SetMtu(1500);
        auto q=CreateObject<DropTailQueue<Packet>>();q->SetAttribute("MaxSize",QueueSizeValue(QueueSize("100p")));
        d->SetQueue(q);node->AddDevice(d);d->SetChannel(channel);return d;
    }
    static bool Valid(const ManagementHeader& h) {
        return h.magic==0x33574131 && !h.reserved && h.kind>=1 && h.kind<=3 && h.channel<4 &&
            h.status<=ManagementHeader::ILLEGAL && h.seq>0 && h.stampNs<=uint64_t(Simulator::Now().GetNanoSeconds());
    }
    static bool ReceiveThunk(ManagementLinks* self,uint32_t node,bool up,Ptr<NetDevice>,
                              Ptr<const Packet> p,uint16_t protocol,const Address&) {
        auto copy=p->Copy();ManagementHeader h;
        if(protocol!=0x88b5||copy->GetSize()<28)return false;
        copy->RemoveHeader(h);
        if(!Valid(h)||h.node!=node||up!=(h.kind!=ManagementHeader::COMMAND)) {
            self->Trace("invalid_rx",node,up,p);return false;
        }
        std::vector<uint8_t> payload(copy->GetSize());copy->CopyData(payload.data(),payload.size());
        self->Trace("rx",node,up,p);
        auto& callback=up?self->coordinator:self->endpoint;
        NS_ABORT_MSG_IF(!callback,"unbound management receiver");callback(h,payload);return true;
    }
    void Trace(const char* event,uint32_t node,bool up,Ptr<const Packet> p) {
        ManagementHeader h;auto copy=p->Copy();copy->RemoveHeader(h);
        std::vector<uint8_t> bytes(copy->GetSize());copy->CopyData(bytes.data(),bytes.size());
        uint64_t hash=14695981039346656037ULL;
        for(auto byte:bytes){hash^=byte;hash*=1099511628211ULL;}
        m_trace<<Simulator::Now().GetNanoSeconds()<<','<<event<<','<<node<<','<<(up?"up":"down")
            <<','<<unsigned(h.kind)<<','<<h.seq<<','<<unsigned(h.channel)<<','<<unsigned(h.status)
            <<','<<h.stampNs<<','<<p->GetSize()<<','<<p->GetUid()<<','<<hash<<'\n';m_trace.flush();
    }
    std::ofstream m_trace;
    std::vector<Ptr<SimpleNetDevice>> m_cn,m_ap;
    std::vector<Ptr<ManagementChannel>> m_channels;
};

// Shared protocol state; actuator callback must finish only after actual readback.
class ManagementEpochs {
public:
    using Done=std::function<void(uint8_t)>;
    using Actuator=std::function<void(uint32_t,uint8_t,uint64_t,Done)>;
    using Readback=std::function<uint8_t(uint32_t)>;
    std::function<void(uint64_t)> reportReady,ackReady;
    ManagementEpochs(ManagementLinks& links,Readback read,Actuator act,const std::string& output)
        :m_links(links),m_read(read),m_act(act),m_trace(output),m_channels(links.Size(),0),
         m_reports(links.Size()),m_reportStamps(links.Size()),m_acks(links.Size()),
         m_apSeq(links.Size()),m_busy(links.Size()),m_payloads(links.Size()),m_reportPayloads(links.Size()) {
        NS_ABORT_MSG_IF(!m_trace,"cannot open management state trace");
        m_trace<<"timeNs,event,seq,known,pending,channels\n";
        links.coordinator=[this](const auto& h,const auto& p){Coordinator(h,p);};
        links.endpoint=[this](const auto& h,const auto&){Endpoint(h);};
    }
    bool Known() const{return m_known;}
    bool Pending() const{return m_pending;}
    const std::vector<uint8_t>& Channels() const{return m_channels;}
    const std::vector<std::vector<uint8_t>>& Payloads() const{return m_payloads;}
    const std::vector<uint8_t>& Statuses() const{return m_statuses;}
    int64_t AvailableAt() const{return m_availableAt;}
    int64_t ObservedAt() const{return m_observedAt;}
    void Reports(uint64_t seq,const std::vector<std::vector<uint8_t>>& payloads={}) {
        NS_ABORT_MSG_IF(!seq||m_pending||(!payloads.empty()&&payloads.size()!=m_links.Size()),"invalid report round");
        for(uint32_t n=0;n<m_links.Size();++n) {
            ManagementHeader h;h.kind=h.REPORT;h.node=n;h.seq=seq;h.channel=m_read(n);
            h.stampNs=Simulator::Now().GetNanoSeconds();
            m_links.Send(h,payloads.empty()?std::vector<uint8_t>{}:payloads[n]);
        }
    }
    bool Submit(uint64_t seq,const std::vector<uint8_t>& targets) {
        if(!m_known||m_pending||seq!=m_epoch||seq<=m_submitted||targets.size()!=m_links.Size()||
            std::any_of(targets.begin(),targets.end(),[](auto c){return c>=4;}))return false;
        m_pending=true;m_known=false;m_submitted=seq;
        std::fill(m_acks.begin(),m_acks.end(),false);m_ackChannels=m_channels;
        State("command_submitted",seq);
        for(uint32_t n=0;n<m_links.Size();++n) {
            ManagementHeader h;h.kind=h.COMMAND;h.node=n;h.seq=seq;h.channel=targets[n];h.stampNs=m_observedAt;
            m_links.Send(h);
        }
        Simulator::Schedule(MilliSeconds(250),&ManagementEpochs::Timeout,this,seq);return true;
    }
private:
    void Coordinator(const ManagementHeader& h,const std::vector<uint8_t>& p) {
        if(h.kind==h.REPORT) {
            if(m_pending||h.seq<=m_epoch||h.seq<m_reportSeq)return;
            if(h.seq>m_reportSeq){m_reportSeq=h.seq;std::fill(m_reports.begin(),m_reports.end(),false);}
            if(m_reports[h.node])return;
            m_reports[h.node]=true;m_reportChannels.resize(m_links.Size());m_reportChannels[h.node]=h.channel;
            m_reportStamps[h.node]=h.stampNs;m_reportPayloads[h.node]=p;
            if(std::all_of(m_reports.begin(),m_reports.end(),[](bool a){return a;})) {
                m_channels=m_reportChannels;m_payloads=m_reportPayloads;m_epoch=h.seq;m_known=true;
                m_observedAt=*std::min_element(m_reportStamps.begin(),m_reportStamps.end());
                m_availableAt=Simulator::Now().GetNanoSeconds();State("report_complete",h.seq);
                if(reportReady)reportReady(h.seq);
            }
        } else if(h.kind==h.ACK) {
            if(!m_pending||h.seq!=m_submitted||m_acks[h.node]){State("ack_ignored",h.seq);return;}
            m_acks[h.node]=true;m_ackChannels[h.node]=h.channel;
            m_ackStatuses.resize(m_links.Size());m_ackStatuses[h.node]=h.status;State("ack_partial",h.seq);
            if(std::all_of(m_acks.begin(),m_acks.end(),[](bool a){return a;})) {
                m_pending=false;m_known=true;m_channels=m_ackChannels;m_statuses=m_ackStatuses;m_availableAt=Simulator::Now().GetNanoSeconds();
                State("ack_complete",h.seq);if(ackReady)ackReady(h.seq);
            }
        }
    }
    void Endpoint(const ManagementHeader& h) {
        if(h.seq<=m_apSeq[h.node]||Simulator::Now().GetNanoSeconds()-int64_t(h.stampNs)>MilliSeconds(200).GetNanoSeconds()) {
            Ack(h,h.STALE);return;
        }
        if(m_busy[h.node]){Ack(h,h.BUSY);return;}
        m_apSeq[h.node]=h.seq;m_busy[h.node]=true;
        m_act(h.node,h.channel,h.seq,[this,h](uint8_t status){
            NS_ABORT_MSG_IF(!m_busy[h.node],"duplicate actuator completion");
            m_busy[h.node]=false;Ack(h,status);
        });
    }
    void Ack(const ManagementHeader& command,uint8_t status) {
        ManagementHeader h;h.kind=h.ACK;h.node=command.node;h.seq=command.seq;
        h.channel=m_read(h.node);h.status=status;h.stampNs=Simulator::Now().GetNanoSeconds();m_links.Send(h);
    }
    void Timeout(uint64_t seq) {
        if(m_pending&&m_submitted==seq){m_pending=false;m_known=false;State("ack_timeout",seq);}
    }
    void State(const char* event,uint64_t seq) {
        m_trace<<Simulator::Now().GetNanoSeconds()<<','<<event<<','<<seq<<','<<m_known<<','<<m_pending<<',';
        for(uint32_t n=0;n<m_channels.size();++n){if(n)m_trace<<';';m_trace<<unsigned(m_channels[n]);}
        m_trace<<'\n';m_trace.flush();
    }
    ManagementLinks& m_links;Readback m_read;Actuator m_act;std::ofstream m_trace;
    std::vector<uint8_t> m_channels,m_reportChannels,m_ackChannels,m_statuses,m_ackStatuses;
    std::vector<bool> m_reports,m_acks,m_busy;
    std::vector<uint64_t> m_reportStamps,m_apSeq;
    std::vector<std::vector<uint8_t>> m_payloads,m_reportPayloads;
    uint64_t m_epoch=0,m_reportSeq=0,m_submitted=0;
    int64_t m_availableAt=-1,m_observedAt=-1;
    bool m_known=false,m_pending=false;
};
} // namespace threeway
