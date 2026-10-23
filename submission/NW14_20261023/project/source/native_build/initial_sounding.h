#pragma once
#include "receiver_sounding.h"
#include <memory>

struct SoundingCosts {
    unsigned actualPulses=0;
    static void Tx(SoundingCosts* costs,ns3::Ptr<const ns3::Packet>) {++costs->actualPulses;}
};

struct InitialSounding {
    std::unique_ptr<ReceiverSounding> receiver;
    std::unique_ptr<SoundingCosts> costs;
};

inline InitialSounding ScheduleInitialSounding(
    const ns3::NodeContainer& stas,
    const std::vector<ns3::Ptr<ns3::WifiNetDevice>>& aps,
    const std::vector<ns3::Ptr<ns3::WifiNetDevice>>& staDevices,
    ns3::Ptr<ns3::SpectrumChannel> channel,const std::vector<uint32_t>& assignment,
    const std::string& output) {
    using namespace ns3;
    const unsigned n=aps.size();
    const double cycle=.002*n+.01,available=.1+4*cycle+.02;
    InitialSounding result;
    result.receiver=std::make_unique<ReceiverSounding>(aps,n,4,available);
    result.costs=std::make_unique<SoundingCosts>();
    auto collector=result.receiver.get();auto costs=result.costs.get();
    for(unsigned i=0;i<n;++i) {
        auto phy=DynamicCast<SpectrumWifiPhy>(aps[i]->GetPhy());
        NS_ABORT_MSG_IF(!phy->TraceConnectWithoutContext("ReceivedPsd",MakeBoundCallback(&ReceiverSounding::Observe,collector,i)),"missing_sounding_received_psd");
    }
    WaveformGeneratorHelper helper;helper.SetChannel(channel);
    helper.SetTxPowerSpectralDensity(WifiSpectrumValueHelper::CreateOfdmTxPowerSpectralDensity(MHz_u{5180},MHz_u{20},DbmToW(dBm_u{16.0206}),MHz_u{20}));
    helper.SetPhyAttribute("Period",TimeValue(MicroSeconds(1000)));
    helper.SetPhyAttribute("DutyCycle",DoubleValue(.123));
    auto emitters=helper.Install(stas);
    std::vector<Ptr<WaveformGenerator>> waves;
    for(unsigned i=0;i<n;++i) {
        auto wave=emitters.Get(i)->GetObject<NonCommunicatingNetDevice>()->GetPhy()->GetObject<WaveformGenerator>();
        NS_ABORT_MSG_IF(!wave->TraceConnectWithoutContext("TxStart",MakeBoundCallback(&SoundingCosts::Tx,costs)),"missing_pilot_tx_trace");
        waves.push_back(wave);
    }
    for(unsigned c=0;c<4;++c) {
        Simulator::Schedule(Seconds(.09+cycle*c),[aps,staDevices,c]() {
            const auto settings=StringValue("{"+std::to_string(36+4*c)+",20,BAND_5GHZ,0}");
            for(unsigned i=0;i<aps.size();++i) {
                NS_ABORT_MSG_IF(!aps[i]->GetPhy()->IsStateIdle() || !staDevices[i]->GetPhy()->IsStateIdle(),"initial_sounding_radio_not_idle");
                aps[i]->GetPhy()->SetAttribute("ChannelSettings",settings);
                staDevices[i]->GetPhy()->SetAttribute("ChannelSettings",settings);
            }
        });
        for(unsigned j=0;j<n;++j) {
            const double start=.1+cycle*c+.002*j;
            collector->AddSlot(j,c,start,start+.002);
            Simulator::Schedule(Seconds(start),[wave=waves[j],c]() {
                wave->SetTxPowerSpectralDensity(WifiSpectrumValueHelper::CreateOfdmTxPowerSpectralDensity(MHz_u{5180.+20*c},MHz_u{20},DbmToW(dBm_u{16.0206}),MHz_u{20}));
                wave->Start();
            });
            Simulator::Schedule(Seconds(start+.0002),&WaveformGenerator::Stop,waves[j]);
        }
    }
    Simulator::Schedule(Seconds(available-.01),[aps,staDevices,assignment]() {
        for(unsigned i=0;i<aps.size();++i) {
            NS_ABORT_MSG_IF(!aps[i]->GetPhy()->IsStateIdle() || !staDevices[i]->GetPhy()->IsStateIdle(),"sounding_restore_radio_not_idle");
            const auto settings=StringValue("{"+std::to_string(36+4*assignment[i])+",20,BAND_5GHZ,0}");
            aps[i]->GetPhy()->SetAttribute("ChannelSettings",settings);
            staDevices[i]->GetPhy()->SetAttribute("ChannelSettings",settings);
        }
    });
    Simulator::Schedule(Seconds(available),[collector,costs,aps,output,n,available]() {
        collector->Write(output+".sounding.csv");
        NS_ABORT_MSG_IF(costs->actualPulses!=n*4,"sounding_actual_transmission_count");
        std::ofstream out(output+".sounding-costs.csv");
        out<<std::setprecision(17)<<"actualPulses,pulseSeconds,nominalTxPowerW,txWaveformEnergyJ,acquisitionStartS,reportAvailableAtS,pilotPayloadBytes\n";
        const double power=static_cast<double>(DbmToW(dBm_u{16.0206}));
        out<<costs->actualPulses<<",0.000123,"<<power<<','<<costs->actualPulses*.000123*power<<",0.09,"<<available<<",0\n";
        out.flush();NS_ABORT_MSG_IF(!out,"sounding_cost_write_failed");
        // Avoid deterministic simultaneous first beacons after sounding.
        for(unsigned i=0;i<aps.size();++i)
            Simulator::Schedule(MilliSeconds(i),[ap=aps[i]] {
                ap->GetMac()->SetAttribute("BeaconGeneration",BooleanValue(true));
            });
    });
    return result;
}
