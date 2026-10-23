#ifndef CORRECTED_TELEMETRY_H
#define CORRECTED_TELEMETRY_H
#include "ns3/wifi-module.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <cmath>
using namespace ns3;

// Causal integration of primary-channel TX/RX/CCA union. Every notification first
// integrates the old horizons only as far as Now; predicted future time is never emitted.
struct CorrectedBusy : WifiPhyListener {
    int64_t last=0, tx=0, rx=0, cca=0, blockedUntil=0;
    bool disabled=false;
    std::vector<std::pair<int64_t,int64_t>> intervals;
    void Advance() {
        int64_t now=Simulator::Now().GetNanoSeconds();
        int64_t a=std::max(last,blockedUntil), b=std::min(now,std::max({tx,rx,cca}));
        if (!disabled && b>a) {
            if (!intervals.empty() && intervals.back().second==a) intervals.back().second=b;
            else intervals.emplace_back(a,b);
        }
        last=now;
    }
    int64_t End(Time d) { return Simulator::Now().GetNanoSeconds()+d.GetNanoSeconds(); }
    void NotifyRxStart(Time d) override { Advance(); cca=last; rx=End(d); }
    void NotifyRxEndOk() override { Advance(); rx=last; }
    void NotifyRxEndError(const WifiTxVector&) override { NotifyRxEndOk(); }
    void NotifyTxStart(Time d,dBm_u) override { Advance(); if(rx<=last) cca=last; rx=last; tx=End(d); }
    void NotifyCcaBusyStart(Time d,WifiChannelListType c,const std::vector<Time>&) override {
        if(c==WIFI_CHANLIST_PRIMARY) { Advance(); cca=End(d); }
    }
    void NotifySwitchingStart(Time d) override { Advance(); tx=rx=cca=last; blockedUntil=End(d); }
    void NotifySleep() override { Advance(); if(rx<=last) cca=last; rx=last; disabled=true; }
    void NotifyOff() override { Advance(); if(rx<=last&&tx<=last) cca=last; tx=rx=last; disabled=true; }
    void NotifyWakeup() override { Advance(); disabled=false; }
    void NotifyOn() override { Advance(); disabled=false; }
    double SecondsIn(int64_t a,int64_t b) const {
        int64_t sum=0;
        for(auto [x,y]:intervals) sum+=std::max<int64_t>(0,std::min(b,y)-std::max(a,x));
        return sum/1e9;
    }
};

struct CorrectedWindow {
    int64_t start=0,end=0,decisionAt=0;
    uint64_t exposureAttempts=0,exposureRetries=0;
    uint64_t completionAttempts=0,completionRetryAttempts=0,terminalOutcomes=0;
    uint64_t failures=0,otherDrops=0,unmatchedTerminalOutcomes=0,leftCensoredTerminals=0;
    uint64_t outstandingMpdus=0,outstandingAttempts=0;
    uint64_t txBytes=0,rxBytes=0,snrCount=0,snrInvalid=0,queuePackets=0,queueCapacity=0;
    double snrDbSum=0,busy=0;
    bool sampled=false,queueValid=true;
    uint64_t observationOpportunities=0,controlOpportunities=0;
};

class CorrectedTelemetry {
public:
    static constexpr const char* estimator="completion_cohort_v1_development";
    struct CompletionRecord {
        std::vector<std::pair<int64_t,bool>> attempts;
        int64_t terminalAt=-1;
        bool beganBeforeObservation=false;
    };
    int64_t start,period,stop;
    std::string output;
    std::vector<std::vector<CorrectedWindow>> windows;
    std::vector<std::shared_ptr<CorrectedBusy>> busy;
    std::vector<Ptr<WifiNetDevice>> devices;
    // UID scoped by STA. Records are retained so a D_k snapshot is independent of
    // same-timestamp callback order and can apply the strict event-time rule t < D_k.
    std::vector<std::map<uint64_t,CompletionRecord>> records;
    CorrectedTelemetry(uint32_t n,double interval,double until,std::string path)
      :start(Seconds(1).GetNanoSeconds()),period(Seconds(interval).GetNanoSeconds()),
       stop(Seconds(until).GetNanoSeconds()),output(path),windows(n),busy(n),devices(n),records(n) {
        NS_ABORT_MSG_IF(period<=0 || stop<=start,"invalid corrected observation window");
        for(uint32_t i=0;i<n;++i) {
            busy[i]=std::make_shared<CorrectedBusy>();
            for(int64_t a=start;a<stop;a+=period) {
                CorrectedWindow w; w.start=a; w.end=std::min(a+period,stop);w.decisionAt=w.end;
                windows[i].push_back(w);
            }
        }
    }
    CorrectedWindow* Current(uint32_t n) {
        auto now=Simulator::Now().GetNanoSeconds();
        if(now<start || now>=stop) return nullptr;
        return &windows.at(n).at((now-start)/period);
    }
    void AddDevice(uint32_t n,Ptr<WifiNetDevice> d) {
        devices.at(n)=d; d->GetPhy()->RegisterListener(busy.at(n));
    }
    void Attempt(uint32_t n,WifiConstPsduMap psdus,WifiTxVector,double) {
        auto w=Current(n);
        const auto now=Simulator::Now().GetNanoSeconds();
        if(now>=stop)return;
        for(const auto& [sta,p]:psdus) for(const auto& mpdu:*p) {
            const auto& h=mpdu->GetHeader();
            if(!h.IsData() || h.GetAddr1().IsGroup())continue;
            if(w) {++w->exposureAttempts; w->exposureRetries+=h.IsRetry();}
            auto& record=records[n][mpdu->GetPacket()->GetUid()];
            NS_ABORT_MSG_IF(record.terminalAt>=0,"attempt after terminal outcome for UID");
            record.beganBeforeObservation=record.beganBeforeObservation||now<start;
            record.attempts.emplace_back(now,h.IsRetry());
        }
    }
    void Outcome(uint32_t n,Ptr<const WifiMpdu> mpdu,bool failed,bool other=false) {
        const auto& header=mpdu->GetHeader();
        if(!header.IsData() || header.GetAddr1().IsGroup())return;
        auto now=Simulator::Now().GetNanoSeconds();
        auto it=records[n].find(mpdu->GetPacket()->GetUid());
        auto w=Current(n);
        if(it==records[n].end() || it->second.attempts.empty() || it->second.terminalAt>=0) {
            if(w)++w->unmatchedTerminalOutcomes;
            return;
        }
        auto& record=it->second;
        record.terminalAt=now;
        if(!w)return; // At D_stop the strict half-open horizon right-censors this result.
        w->completionAttempts+=record.attempts.size();
        w->completionRetryAttempts+=std::count_if(record.attempts.begin(),record.attempts.end(),
                                                   [](const auto& event){return event.second;});
        ++w->terminalOutcomes;w->failures+=failed;w->otherDrops+=other;
        w->leftCensoredTerminals+=record.beganBeforeObservation;
    }
    void Ack(uint32_t n,Ptr<const WifiMpdu> p) { Outcome(n,p,false); }
    void Drop(uint32_t n,WifiMacDropReason r,Ptr<const WifiMpdu> p) {
        Outcome(n,p,r==WIFI_MAC_DROP_REACHED_RETRY_LIMIT,r!=WIFI_MAC_DROP_REACHED_RETRY_LIMIT);
    }
    void Rx(uint32_t n,double snr) { if(auto w=Current(n)) {
        if(std::isfinite(snr)&&snr>0) {w->snrDbSum+=10*std::log10(snr);++w->snrCount;}
        else ++w->snrInvalid;
    }}
    void App(uint32_t n,Ptr<const Packet> p,bool received) {if(auto w=Current(n)) {
        if(received) w->rxBytes+=p->GetSize();else w->txBytes+=p->GetSize();
    }}
    void ControlOpportunity(uint32_t n) {
        auto w=Current(n); NS_ABORT_MSG_IF(!w,"control opportunity outside measurement horizon");
        ++w->controlOpportunities;
    }
    void Snapshot(size_t index,bool opportunity) {
        for(uint32_t n=0;n<windows.size();++n) {
            auto& w=windows[n].at(index);if(w.sampled)continue;
            busy[n]->Advance();w.busy=busy[n]->SecondsIn(w.start,w.end);
            w.sampled=true;w.observationOpportunities=opportunity;
            // Strict endpoint capture: outcomes timestamped exactly D_k remain
            // outstanding for decision D_k even if their callback ran first.
            for(const auto& [uid,record]:records[n]) {
                (void)uid;
                uint64_t attemptsBefore=std::count_if(record.attempts.begin(),record.attempts.end(),
                    [&](const auto& event){return event.first<w.end;});
                if(attemptsBefore && (record.terminalAt<0 || record.terminalAt>=w.end)) {
                    ++w.outstandingMpdus;w.outstandingAttempts+=attemptsBefore;
                }
            }
            if(!devices[n]) {w.queueValid=false;continue;}
            auto capture=[&](Ptr<WifiMacQueue> q) {
                if(!q || q->GetMaxSize().GetUnit()!=QueueSizeUnit::PACKETS) {w.queueValid=false;return;}
                w.queuePackets+=q->GetNPackets();w.queueCapacity+=q->GetMaxSize().GetValue();
            };
            auto mac=devices[n]->GetMac();
            if(mac->GetQosSupported())for(auto ac:{AC_BE,AC_BK,AC_VI,AC_VO})capture(mac->GetTxopQueue(ac));
            else capture(mac->GetTxop()->GetWifiMacQueue());
            w.queueValid=w.queueValid&&w.queueCapacity>0&&w.queuePackets<=w.queueCapacity;
        }
    }
    void Start() {
        for(size_t j=0;j<windows[0].size();++j) {
            auto& w=windows[0][j];
            // Observation schedule only, never an eligible channel-control opportunity.
            if(w.end<stop) Simulator::Schedule(NanoSeconds(w.end),&CorrectedTelemetry::Snapshot,this,j,true);
        }
    }
    void Write() {
        auto now=Simulator::Now().GetNanoSeconds();
        NS_ABORT_MSG_IF(now!=stop,"corrected finalization must occur exactly at declared stop");
        Snapshot(windows[0].size()-1,false);
        std::ofstream o(output);NS_ABORT_MSG_IF(!o,"cannot open corrected telemetry");o<<std::setprecision(17);
        o<<"estimator,node,windowStartNs,windowEndNs,decisionAtNs,exposureAttempts,exposureRetryAttempts,completionAttempts,completionRetryAttempts,terminalOutcomes,retryLimitFailures,otherTerminalDrops,unmatchedTerminalOutcomes,leftCensoredTerminalOutcomes,outstandingMpdusAtDecision,outstandingAttemptsAtDecision,completionCountsValid,busySeconds,busyValid,snrMeanDb,snrSamples,snrInvalid,snrValid,queuePackets,queueCapacityPackets,queueValid,appTxBytes,appRxBytes,offeredBps,deliveredBps,observationOpportunities,controlOpportunities\n";
        for(uint32_t n=0;n<windows.size();++n)for(auto& w:windows[n]) {
            const double elapsed=(w.end-w.start)/1e9;
            const bool valid=w.otherDrops==0&&w.unmatchedTerminalOutcomes==0&&w.leftCensoredTerminals==0&&
                             (w.terminalOutcomes>0||w.outstandingMpdus==0);
            o<<estimator<<','<<n<<','<<w.start<<','<<w.end<<','<<w.decisionAt<<','<<w.exposureAttempts<<','
             <<w.exposureRetries<<','<<w.completionAttempts<<','<<w.completionRetryAttempts<<','
             <<w.terminalOutcomes<<','<<w.failures<<','<<w.otherDrops<<','
             <<w.unmatchedTerminalOutcomes<<','<<w.leftCensoredTerminals<<','<<w.outstandingMpdus<<','
             <<w.outstandingAttempts<<','<<valid<<','<<w.busy<<','<<w.sampled<<',';
            if(w.snrCount&&w.snrInvalid==0)o<<w.snrDbSum/w.snrCount;
            o<<','<<w.snrCount<<','<<w.snrInvalid<<','<<(w.snrCount>0&&w.snrInvalid==0)<<','<<w.queuePackets<<','<<w.queueCapacity<<','<<w.queueValid<<','<<w.txBytes<<','<<w.rxBytes<<','<<w.txBytes*8/elapsed<<','<<w.rxBytes*8/elapsed<<','<<w.observationOpportunities<<','<<w.controlOpportunities<<'\n';
        }
    }
};
static CorrectedTelemetry* g_corrected=nullptr;
static void CorrectedAttempt(uint32_t n,WifiConstPsduMap p,WifiTxVector v,double power){g_corrected->Attempt(n,p,v,power);}
static void CorrectedAck(uint32_t n,Ptr<const WifiMpdu> p){g_corrected->Ack(n,p);}
static void CorrectedDrop(uint32_t n,WifiMacDropReason r,Ptr<const WifiMpdu> p){g_corrected->Drop(n,r,p);}
#endif
