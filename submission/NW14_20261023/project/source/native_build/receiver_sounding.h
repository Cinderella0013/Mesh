#pragma once
#include "ns3/spectrum-module.h"
#include "ns3/wifi-module.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

// Only receiver-local ReceivedPsd and a preannounced exclusive pilot schedule.
// No geometry, txPhy, senderNodeId or simulator-known failed frame is inspected.
class ReceiverSounding {
public:
    struct Slot { unsigned sender, channel; ns3::Time start, end; };
    struct Cell { unsigned arrivals=0; double power=0, arrival=0, pulseEnd=0; bool invalid=false; };
    ReceiverSounding(std::vector<ns3::Ptr<ns3::WifiNetDevice>> receivers,
                     unsigned senders, unsigned channels, double available)
        : devices(std::move(receivers)), n(senders), k(channels), availableAt(available),
          cells(devices.size()*n*k) {
        if(!n || !k || k>4 || !std::isfinite(availableAt))throw std::invalid_argument("invalid_sounding_config");
    }
    void AddSlot(unsigned sender,unsigned channel,double start,double end) {
        if(!std::isfinite(start) || !std::isfinite(end))throw std::invalid_argument("invalid_pilot_slot");
        const auto begins=ns3::Seconds(start),ends=ns3::Seconds(end);
        if(sender>=n || channel>=k || !(ns3::Seconds(0)<=begins && begins<ends && ends<=ns3::Seconds(availableAt)))
            throw std::invalid_argument("invalid_pilot_slot");
        for(const auto& s:slots)
            if((s.sender==sender && s.channel==channel) || (begins<s.end && s.start<ends))
                throw std::invalid_argument("duplicate_or_overlapping_pilot_slot");
        slots.push_back({sender,channel,begins,ends});
    }
    static void Observe(ReceiverSounding* self,unsigned receiver,
                        ns3::Ptr<const ns3::SpectrumValue> psd,ns3::Time duration) {
        if(duration.GetNanoSeconds()!=123000)return; // known controlled pilot signature
        const auto now=ns3::Simulator::Now();
        const Slot* current=nullptr;
        for(const auto& slot:self->slots)if(slot.start<=now && now<slot.end)current=&slot;
        if(!current)return;
        auto& cell=self->cells.at((receiver*self->n+current->sender)*self->k+current->channel);
        ++cell.arrivals; cell.arrival=now.GetSeconds(); cell.pulseEnd=(now+duration).GetSeconds();
        if(!psd || cell.arrivals!=1 || now+duration>current->end ||
           self->devices.at(receiver)->GetPhy()->GetChannelNumber()!=36+4*current->channel) {
            cell.invalid=true; return;
        }
        const double center=(5180.+20*current->channel)*1e6;
        double power=0;
        auto band=psd->ConstBandsBegin(); auto value=psd->ConstValuesBegin();
        for(;band!=psd->ConstBandsEnd();++band,++value) {
            if(!std::isfinite(*value) || *value<0 || !std::isfinite(band->fl) ||
               !std::isfinite(band->fh) || band->fh<=band->fl) {cell.invalid=true;return;}
            const double width=std::max(0.,std::min(band->fh,center+10e6)-std::max(band->fl,center-10e6));
            power+=*value*width;
        }
        if(!std::isfinite(power) || power<=0)cell.invalid=true;
        else cell.power=power;
    }
    void Write(const std::string& path,uint64_t batchEpoch=0) const {
        if(ns3::Simulator::Now().GetSeconds()<availableAt)throw std::runtime_error("report_not_available");
        if(slots.size()!=n*k)throw std::runtime_error("incomplete_pilot_schedule");
        std::ofstream out(path);
        if(!out)throw std::runtime_error("cannot_write_receiver_report");
        out<<std::setprecision(17);
        out<<"receiver,sender,channelIndex,slotStartS,slotEndS,arrivalS,pulseEndS,availableAtS,meanPowerW,valid,reason";
        if(batchEpoch)out<<",batchEpoch";out<<'\n';
        for(unsigned receiver=0;receiver<devices.size();++receiver)for(const auto& slot:slots) {
            const auto& cell=cells.at((receiver*n+slot.sender)*k+slot.channel);
            const bool valid=cell.arrivals==1&&!cell.invalid;
            out<<receiver<<','<<slot.sender<<','<<slot.channel<<','<<slot.start.GetSeconds()<<','<<slot.end.GetSeconds()<<',';
            if(cell.arrivals)out<<cell.arrival;
            out<<',';if(cell.arrivals)out<<cell.pulseEnd;
            out<<','<<availableAt<<',';if(valid)out<<cell.power;
            out<<','<<(valid?"true":"false")<<','<<(valid?"received_psd_pilot":cell.arrivals?"invalid_or_duplicate_pilot":"missing_pilot");
            if(batchEpoch)out<<','<<batchEpoch;out<<'\n';
        }
        out.flush();if(!out)throw std::runtime_error("receiver_report_write_failed");
    }
    std::vector<double> Row(unsigned receiver) const {
        if(ns3::Simulator::Now()<ns3::Seconds(availableAt))throw std::runtime_error("report_not_available");
        if(receiver>=devices.size()||slots.size()!=n*k)throw std::runtime_error("invalid_receiver_row");
        std::vector<double> result;
        for(unsigned sender=0;sender<n;++sender)for(unsigned channel=0;channel<k;++channel) {
            const auto& cell=cells.at((receiver*n+sender)*k+channel);
            if(cell.arrivals!=1||cell.invalid||!std::isfinite(cell.power)||cell.power<=0)
                throw std::runtime_error("unavailable_receiver_cell");
            result.push_back(cell.power);
        }
        return result;
    }
private:
    std::vector<ns3::Ptr<ns3::WifiNetDevice>> devices;
    unsigned n,k;
    double availableAt;
    std::vector<Cell> cells;
    std::vector<Slot> slots;
};
