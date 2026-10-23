#ifndef THREEWAY_QUEUE_CONTINUITY_H
#define THREEWAY_QUEUE_CONTINUITY_H
#include "ns3/wifi-module.h"
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <vector>
using namespace ns3;

// Evaluator only. Observe the native queue; never dequeue, peek (which can
// expire an MPDU), add a packet tag, retime traffic or supply policy features.
class QueueContinuity {
  public:
    explicit QueueContinuity(const std::string& path):m_output(path) {
        NS_ABORT_MSG_IF(!m_output,"cannot open queue continuity trace");
        m_output<<"timeNs,node,role,queue,event,packetUid,packetBytes,trackedPackets,actualPackets,trackedBytes,actualBytes,epoch,phase,queuedUids,inflightUids\n";
    }
    void AddDevice(uint32_t node,const char* role,Ptr<WifiNetDevice> device) {
        auto mac=device->GetMac();
        if(auto txop=mac->GetTxop())AddQueue(node,role,"nonqos",txop->GetWifiMacQueue());
        for(auto ac:{AC_BE,AC_BK,AC_VI,AC_VO})
            if(auto queue=mac->GetTxopQueue(ac))AddQueue(node,role,std::to_string(ac),queue);
    }
    void Checkpoint(uint32_t node,uint64_t epoch,const char* phase) {
        NS_ABORT_MSG_IF(m_finished,"queue evidence already closed");
        for(auto& entry:m_entries)if(entry.node==node) {
            Verify(entry);std::ostringstream queued,inflight;bool first=true,firstFlight=true;
            for(const auto& [uid,mpdu]:entry.members) {
                NS_ABORT_MSG_IF(!mpdu->IsQueued(),"tracked MPDU is no longer queued");
                if(!first)queued<<'|';queued<<uid;first=false;
                if(mpdu->IsInFlight()) {if(!firstFlight)inflight<<'|';inflight<<uid;firstFlight=false;}
            }
            Write(entry,"checkpoint",nullptr,epoch,phase,queued.str(),inflight.str());
        }
    }
    void Finish() {
        std::set<uint32_t> nodes;for(auto& entry:m_entries)nodes.insert(entry.node);
        for(auto node:nodes)Checkpoint(node,0,"horizon");
        m_output.flush();NS_ABORT_MSG_IF(!m_output,"queue evidence write failed");m_finished=true;
    }
  private:
    struct Entry {
        uint32_t node;std::string role,name;Ptr<WifiMacQueue> queue;
        std::map<uint64_t,Ptr<const WifiMpdu>> members;
    };
    void AddQueue(uint32_t node,const char* role,std::string name,Ptr<WifiMacQueue> queue) {
        for(const auto& entry:m_entries)if(entry.queue==queue)return;
        NS_ABORT_MSG_IF(queue->GetNPackets()||queue->GetNBytes(),"observer must attach before queue use");
        const uint32_t index=m_entries.size();m_entries.push_back({node,role,std::move(name),queue,{}});
        for(const auto& [trace,event]:std::vector<std::pair<const char*,const char*>>{
            {"Enqueue","enqueue"},{"Dequeue","dequeue"},{"DropBeforeEnqueue","drop_before"},
            {"DropAfterDequeue","drop_after"},{"Expired","expired"}})
            NS_ABORT_MSG_IF(!queue->TraceConnectWithoutContext(trace,
                MakeBoundCallback(&QueueContinuity::Event,this,index,event)),"missing native queue trace "<<trace);
    }
    static void Event(QueueContinuity* self,uint32_t index,const char* event,Ptr<const WifiMpdu> mpdu) {
        if(self->m_finished)return; // Simulator::Destroy is outside the measured horizon.
        auto& entry=self->m_entries.at(index);const auto uid=mpdu->GetPacket()->GetUid();
        if(std::string(event)=="enqueue")
            NS_ABORT_MSG_IF(!entry.members.emplace(uid,mpdu).second,"duplicate UID within native queue");
        else if(std::string(event)=="dequeue")
            NS_ABORT_MSG_IF(entry.members.erase(uid)!=1,"unobserved native queue departure");
        // Expired can be scheduled before/after the actual dequeue. A remove
        // emits Dequeue then DropAfterDequeue: the latter never removes twice.
        self->Verify(entry);self->Write(entry,event,mpdu,0,"","","");
    }
    static uint64_t Bytes(const Entry& entry) {
        uint64_t bytes=0;for(const auto& [uid,mpdu]:entry.members)bytes+=mpdu->GetSize();return bytes;
    }
    static void Verify(const Entry& entry) {
        NS_ABORT_MSG_IF(entry.members.size()!=entry.queue->GetNPackets()||Bytes(entry)!=entry.queue->GetNBytes(),
                        "native queue count/bytes disagree with UID trace");
    }
    void Write(const Entry& entry,const char* event,Ptr<const WifiMpdu> mpdu,uint64_t epoch,
               const char* phase,const std::string& queued,const std::string& inflight) {
        m_output<<Simulator::Now().GetNanoSeconds()<<','<<entry.node<<','<<entry.role<<','<<entry.name<<','<<event<<',';
        if(mpdu)m_output<<mpdu->GetPacket()->GetUid();m_output<<',';
        if(mpdu)m_output<<mpdu->GetSize();
        m_output<<','<<entry.members.size()<<','<<entry.queue->GetNPackets()<<','<<Bytes(entry)<<','
                <<entry.queue->GetNBytes()<<','<<epoch<<','<<phase<<','<<queued<<','<<inflight<<'\n';
        NS_ABORT_MSG_IF(!m_output,"queue trace write failed");
    }
    std::ofstream m_output;std::vector<Entry> m_entries;bool m_finished=false;
};
#endif
