#pragma once
#include "initial_sounding.h"
#include <functional>

struct SoundingSchedule {
    explicit SoundingSchedule(bool compact):compact(compact) {}
    int CycleUs() const {return compact?5000:40000;}
    int SlotUs() const {return compact?250:2000;}
    int FirstPilotUs() const {return compact?2000:3000;}
    int RestoreUs() const {return compact?21000:161000;}
    int FinishUs() const {return compact?22000:163000;}
    const bool compact;
};

// Diagnostic receiver batches only; no geometry or future action outcome enters reports.
class FreshSounding {
public:
    FreshSounding(const ns3::NodeContainer& stas,std::vector<ns3::Ptr<ns3::WifiNetDevice>> aps,
                  ns3::Ptr<ns3::SpectrumChannel> channel,std::string output,bool compact=false)
        :m_aps(std::move(aps)),m_output(std::move(output)),m_cost(m_output+".fresh-costs.csv"),m_schedule(compact) {
        using namespace ns3;
        NS_ABORT_MSG_IF(!m_cost,"fresh cost trace unavailable");
        m_cost<<"epoch,beginNs,availableNs,actualPulses,txEnergyJ\n";
        WaveformGeneratorHelper helper;helper.SetChannel(channel);
        helper.SetTxPowerSpectralDensity(WifiSpectrumValueHelper::CreateOfdmTxPowerSpectralDensity(MHz_u{5180},MHz_u{20},DbmToW(dBm_u{16.0206}),MHz_u{20}));
        helper.SetPhyAttribute("Period",TimeValue(MicroSeconds(1000)));
        helper.SetPhyAttribute("DutyCycle",DoubleValue(.123));
        auto emitters=helper.Install(stas);
        for(unsigned n=0;n<m_aps.size();++n) {
            auto wave=emitters.Get(n)->GetObject<NonCommunicatingNetDevice>()->GetPhy()->GetObject<WaveformGenerator>();
            NS_ABORT_MSG_IF(!wave->TraceConnectWithoutContext("TxStart",MakeBoundCallback(&SoundingCosts::Tx,&m_pulses)),"pilot TX trace absent");
            m_waves.push_back(wave);
            auto phy=DynamicCast<SpectrumWifiPhy>(m_aps[n]->GetPhy());
            NS_ABORT_MSG_IF(!phy->TraceConnectWithoutContext("ReceivedPsd",MakeBoundCallback(&FreshSounding::Observe,this,n)),"fresh PSD trace absent");
        }
    }
    void Begin(uint64_t epoch) {
        using namespace ns3;
        NS_ABORT_MSG_IF(m_active,"overlapping fresh batch");
        m_epoch=epoch;m_begin=Simulator::Now();m_pulses.actualPulses=0;
        m_active=std::make_unique<ReceiverSounding>(m_aps,m_aps.size(),4,(m_begin+MicroSeconds(m_schedule.FinishUs())).GetSeconds());
        for(unsigned c=0;c<4;++c)for(unsigned j=0;j<m_aps.size();++j) {
            const Time offset=MicroSeconds(m_schedule.FirstPilotUs()+m_schedule.CycleUs()*c+m_schedule.SlotUs()*j);
            m_active->AddSlot(j,c,(m_begin+offset).GetSeconds(),(m_begin+offset+MicroSeconds(m_schedule.SlotUs())).GetSeconds());
            Simulator::Schedule(offset,[wave=m_waves[j],c] {
                wave->SetTxPowerSpectralDensity(WifiSpectrumValueHelper::CreateOfdmTxPowerSpectralDensity(MHz_u{5180.+20*c},MHz_u{20},DbmToW(dBm_u{16.0206}),MHz_u{20}));
                wave->Start();
            });
            Simulator::Schedule(offset+MicroSeconds(200),&WaveformGenerator::Stop,m_waves[j]);
        }
    }
    const SoundingSchedule& Schedule() const {return m_schedule;}
    const std::vector<double>& Row(unsigned receiver) const {return m_rows.at(receiver);}
    uint64_t Epoch() const {return m_epoch;}
    int64_t OldestNs() const {return (m_begin+ns3::MicroSeconds(m_schedule.FirstPilotUs())).GetNanoSeconds();}
    int64_t ReadyNs() const {return (m_begin+ns3::MicroSeconds(m_schedule.FinishUs())).GetNanoSeconds();}
    std::function<void(uint64_t)> reportReady;
    std::function<void(uint64_t)> reportUnavailable;
    void Unavailable(uint64_t epoch) {if(reportUnavailable)reportUnavailable(epoch);}
    void Publish() {if(reportReady)reportReady(m_epoch);}
    void Finish() {
        using namespace ns3;
        NS_ABORT_MSG_IF(!m_active||m_pulses.actualPulses!=60,"incomplete transmitted pilot batch");
        m_active->Write(m_output+".fresh-"+std::to_string(m_epoch)+".csv",m_epoch);
        m_rows.clear();
        for(unsigned n=0;n<m_aps.size();++n)m_rows.push_back(m_active->Row(n));
        m_cost<<std::setprecision(17)<<m_epoch<<','<<m_begin.GetNanoSeconds()<<','<<Simulator::Now().GetNanoSeconds()<<','
            <<m_pulses.actualPulses<<','<<m_pulses.actualPulses*.000123*static_cast<double>(DbmToW(dBm_u{16.0206}))<<'\n';
        m_cost.flush();NS_ABORT_MSG_IF(!m_cost,"fresh cost write failed");m_active.reset();
    }
private:
    static void Observe(FreshSounding* self,unsigned receiver,ns3::Ptr<const ns3::SpectrumValue> psd,ns3::Time duration) {
        if(self->m_active)ReceiverSounding::Observe(self->m_active.get(),receiver,psd,duration);
    }
    std::vector<ns3::Ptr<ns3::WifiNetDevice>> m_aps;
    std::vector<ns3::Ptr<ns3::WaveformGenerator>> m_waves;
    std::unique_ptr<ReceiverSounding> m_active;
    std::vector<std::vector<double>> m_rows;
    std::string m_output;std::ofstream m_cost;SoundingCosts m_pulses;ns3::Time m_begin;uint64_t m_epoch=0;
    const SoundingSchedule m_schedule;
};
