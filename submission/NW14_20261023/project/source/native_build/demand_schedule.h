#pragma once
#include "ns3/core-module.h"
#include <charconv>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <sstream>
#include <vector>
using namespace ns3;

// Prospective continuous-positive offered rate. The policy sees At(Now) only;
// integral lookup is retrospective reward/evaluator bookkeeping, not a feature.
class DemandSchedule {
public:
    int64_t start=Seconds(1).GetNanoSeconds(),stop,until;
    uint32_t nodes;
    std::map<int64_t,std::vector<uint64_t>> steps;
    DemandSchedule(uint32_t n,double simulationStop,const std::string& filename)
        :stop(Seconds(simulationStop).GetNanoSeconds()),until(stop-MilliSeconds(100).GetNanoSeconds()),nodes(n) {
        NS_ABORT_MSG_IF(!n||until<=start,"invalid demand horizon");
        std::ifstream input(filename);NS_ABORT_MSG_IF(!input,"cannot read prospective demand schedule");
        std::string line;std::getline(input,line);if(!line.empty()&&line.back()=='\r')line.pop_back();
        NS_ABORT_MSG_IF(line!="timeNs,node,nominalBps","demand CSV schema mismatch");
        int64_t previous=-1;uint32_t expectedNode=0;
        while(std::getline(input,line)) {
            if(!line.empty()&&line.back()=='\r')line.pop_back();
            std::vector<std::string> fields;std::istringstream stream(line);std::string field;
            while(std::getline(stream,field,','))fields.push_back(field);
            NS_ABORT_MSG_IF(fields.size()!=3||line.empty()||line.back()==',',"invalid demand CSV row");
            const auto time=Unsigned(fields[0]),node=Unsigned(fields[1]),rate=Unsigned(fields[2]);
            NS_ABORT_MSG_IF(time<uint64_t(start)||time>=uint64_t(until)||node>=n||rate==0||rate>1000000000,
                            "demand time/node/rate outside qualified domain");
            if(int64_t(time)!=previous) {
                NS_ABORT_MSG_IF((previous<0&&time!=uint64_t(start))||(previous>=0&&(expectedNode!=n||int64_t(time)<=previous)),
                                "demand snapshots must be complete and strictly ordered from1s");
                previous=time;expectedNode=0;steps[previous].resize(n);
            }
            NS_ABORT_MSG_IF(node!=expectedNode++,"ordered complete demand nodes required");
            steps[previous][node]=rate;
        }
        NS_ABORT_MSG_IF(steps.empty()||expectedNode!=n,"missing complete demand snapshot");
    }
    static uint64_t Unsigned(const std::string& text) {
        uint64_t value=0;auto result=std::from_chars(text.data(),text.data()+text.size(),value);
        NS_ABORT_MSG_IF(text.empty()||result.ec!=std::errc()||result.ptr!=text.data()+text.size(),"unsigned demand integer required");
        return value;
    }
    uint64_t At(uint32_t node,int64_t time) const {
        NS_ABORT_MSG_IF(node>=nodes||time<start||time>stop,"demand lookup outside horizon");
        return std::prev(steps.upper_bound(time))->second.at(node);
    }
    double Bytes(uint32_t node,int64_t a,int64_t b) const {
        NS_ABORT_MSG_IF(node>=nodes||a<start||b<a||b>stop,"invalid retrospective demand window");
        double bytes=0;
        for(auto it=steps.begin();it!=steps.end();++it) {
            const auto following=std::next(it);
            const int64_t end=following==steps.end()?until:following->first;
            const int64_t elapsed=std::max<int64_t>(0,std::min({b,end,until})-std::max(a,it->first));
            // Bounds ensure the integer product fits uint64 before conversion.
            bytes+=uint64_t(elapsed)*it->second[node]/8e9;
        }
        return bytes;
    }
    void Start(std::function<void(uint32_t,uint64_t)> apply) const {
        for(const auto& [time,rates]:steps)Simulator::Schedule(NanoSeconds(time),[apply,rates]{
            for(uint32_t n=0;n<rates.size();++n)apply(n,rates[n]);
        });
    }
    void Write(double interval,const std::string& path) const {
        NS_ABORT_MSG_IF(Simulator::Now().GetNanoSeconds()!=stop,"demand closes at declared simulation stop");
        std::ofstream output(path);NS_ABORT_MSG_IF(!output,"cannot write demand windows");
        output<<std::setprecision(17)<<"node,windowStartNs,windowEndNs,nominalBytes\n";
        const int64_t period=Seconds(interval).GetNanoSeconds();NS_ABORT_MSG_IF(period<=0,"positive demand window required");
        for(uint32_t n=0;n<nodes;++n)for(int64_t a=start;a<stop;a+=period) {
            const auto b=std::min(a+period,stop);output<<n<<','<<a<<','<<b<<','<<Bytes(n,a,b)<<'\n';
        }
        output.flush();NS_ABORT_MSG_IF(!output,"cannot finalize demand windows");
    }
};
