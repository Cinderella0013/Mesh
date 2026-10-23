#include "ns3/channel-access-manager.h"
#include "observed_energy_window.h"
#include "corrected_telemetry.h"
#include "recovery_domain.h"
#include "v33_hybrid_execution.h"
#include "initial_sounding.h"
#include "fresh_sounding.h"
#include "policy_bridge.h"
#include "packet_cohort.h"
#include "queue_continuity.h"
#include "burst_demand.h"
#include "scan_gate.h"
#include "tracking_channel.h"

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/onoff-application.h"
#include "ns3/spectrum-module.h"
#include "ns3/wifi-module.h"
#include "ns3/wifi-utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <sched.h>
#include <unistd.h>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("WifiMeshPacketValidation");

// Evaluator-only diagnostic: never connected to PolicyBridge or an action.
struct RobustnessAudit {
    std::ofstream radio, heard, trajectory, attributes, begins, association, emitter;
    std::vector<Mac48Address> addresses;
    RobustnessAudit(const std::string& p):radio(p+".probe-radio.csv"),heard(p+".probe-heard.csv"),
        trajectory(p+".probe-motion.csv"),attributes(p+".probe-phy.csv"),begins(p+".probe-rxbegin.csv"),
        association(p+".association.csv"),emitter(p+".emitter-arrival.csv") {
        NS_ABORT_MSG_IF(!radio||!heard||!trajectory||!attributes||!begins,"probe output unavailable");
        radio<<std::setprecision(17)<<"timeS,receiverRole,receiver,senderNode,rxPowerDbm,durationS\n";
        heard<<"timeS,receiver,sender,decodedData\n";
        trajectory<<std::setprecision(17)<<"timeS,node,x,y,z,vx,vy,associated,channel\n";
        attributes<<"node,name,valueDbm\n";
        begins<<"timeS,receiver,sourceAddress,frameType\n";
        NS_ABORT_MSG_IF(!association||!emitter,"association diagnostic output unavailable");
        association<<"timeNs,node,event,bssid\n";
        emitter<<std::setprecision(17)<<"timeNs,role,node,powerDbm,durationNs,rxCenterMHz\n";
    }
    static void Association(RobustnessAudit* self,unsigned node,std::string event,Mac48Address bssid) {
        self->association<<Simulator::Now().GetNanoSeconds()<<','<<node<<','<<event<<','<<bssid<<'\n';
    }
    static void Beacon(RobustnessAudit* self,unsigned node,Time at) {
        self->association<<at.GetNanoSeconds()<<','<<node<<",beacon,\n";
    }
    static void Emitter(RobustnessAudit* self,std::string role,unsigned node,
        Ptr<const SpectrumSignalParameters> signal,uint32_t sender,double dbm,Time duration) {
        if(sender==30) {
            auto first=signal->psd->ConstBandsBegin(),last=signal->psd->ConstBandsEnd();--last;
            self->emitter<<Simulator::Now().GetNanoSeconds()<<','<<role<<','<<node<<','<<dbm<<','<<duration.GetNanoSeconds()<<','<<(first->fc+last->fc)/2e6<<'\n';
        }
    }
    static void Arrival(RobustnessAudit* self,std::string role,unsigned receiver,
        Ptr<const SpectrumSignalParameters> signal,uint32_t sender,double dbm,Time duration) {
        if(sender!=0 && sender!=1 && sender!=15 && sender!=16 && sender!=30)return;
        self->radio<<Simulator::Now().GetSeconds()<<','<<role<<','<<receiver<<','<<sender<<','<<dbm<<','<<duration.GetSeconds()<<'\n';
    }
    static void Heard(RobustnessAudit* self,unsigned receiver,Ptr<const Packet> packet,
        double snr,WifiMode mode,WifiPreamble preamble) {
        if(Simulator::Now()<Seconds(1))return;
        WifiMacHeader h;packet->PeekHeader(h);
        if(!h.IsData())return;
        for(unsigned sender=0;sender<2;++sender)if(sender!=receiver && h.GetAddr2()==self->addresses.at(sender))
            self->heard<<Simulator::Now().GetSeconds()<<','<<receiver<<','<<sender<<",1\n";
    }
    static void Begin(RobustnessAudit* self,unsigned receiver,Ptr<const Packet> packet,RxPowerWattPerChannelBand powers) {
        WifiMacHeader h;packet->PeekHeader(h);
        if(h.IsData()||h.IsMgt())self->begins<<Simulator::Now().GetSeconds()<<','<<receiver<<','<<h.GetAddr2()<<','<<unsigned(h.GetType())<<'\n';
    }
    void Sample(const NodeContainer& stas,const std::vector<Ptr<WifiNetDevice>>& devices) {
        for(unsigned n=0;n<stas.GetN();++n) {
            auto mobility=stas.Get(n)->GetObject<MobilityModel>();auto p=mobility->GetPosition();auto v=mobility->GetVelocity();
            auto mac=DynamicCast<StaWifiMac>(devices[n]->GetMac());
            trajectory<<Simulator::Now().GetSeconds()<<','<<n<<','<<p.x<<','<<p.y<<','<<p.z<<','<<v.x<<','<<v.y<<','<<mac->IsAssociated()<<','<<unsigned(devices[n]->GetPhy()->GetChannelNumber())<<'\n';
        }
    }
};


struct TelemetryWindow
{
    uint64_t appTxPackets{0};
    uint64_t appTxBytes{0};
    uint64_t appRxPackets{0};
    uint64_t appRxBytes{0};
    uint64_t macTxPackets{0};
    uint64_t retryEvents{0};
    uint64_t finalFailures{0};
    uint64_t phyRxOk{0};
    uint64_t phyRxError{0};
    double snrSum{0.0};
    double snrSquaredSum{0.0};
    double busySeconds{0.0};
};

static double
Clamp01(double value)
{
    return std::max(0.0, std::min(1.0, value));
}

static const char*
RiskClassName(v33::RiskClass riskClass)
{
    switch (riskClass)
    {
    case v33::RiskClass::Normal:
        return "normal";
    case v33::RiskClass::Collision:
        return "collision";
    case v33::RiskClass::Capacity:
        return "capacity";
    }
    return "unknown";
}

// Offline-only A5 audit recorder.  It never contributes a runtime feature or controller input.
class OfflineEnergyRecorder
{
  public:
    OfflineEnergyRecorder(uint32_t numNodes, uint32_t emitterNodeId, const std::string& outputPath)
        : m_totals(numNodes), m_devices(numNodes), m_emitterNodeId(emitterNodeId), m_outputPath(outputPath)
    {
    }

    void AddDevice(uint32_t node, Ptr<WifiNetDevice> device)
    {
        m_devices.at(node) = device;
    }

    void SignalArrival(uint32_t node,
                       Ptr<const SpectrumSignalParameters> signal,
                       uint32_t senderNodeId,
                       double,
                       Time duration)
    {
        if (senderNodeId != m_emitterNodeId || !signal || !signal->psd)
        {
            return;
        }
        const auto device = m_devices.at(node);
        NS_ABORT_MSG_IF(!device, "missing probe device for STA " << node);
        const double centerHz = (5000.0 + 5.0 * device->GetPhy()->GetChannelNumber()) * 1e6;
        const double wantedLowHz = centerHz - 10e6;
        const double wantedHighHz = centerHz + 10e6;
        const double adjacentLowHz = centerHz - 30e6;
        const double adjacentHighHz = centerHz + 30e6;
        auto band = signal->psd->ConstBandsBegin();
        auto value = signal->psd->ConstValuesBegin();
        for (; band != signal->psd->ConstBandsEnd(); ++band, ++value)
        {
            const double wattsPerHz = *value;
            if (!std::isfinite(wattsPerHz) || wattsPerHz < 0.0)
            {
                continue;
            }
            const double wantedHz = Overlap(band->fl, band->fh, wantedLowHz, wantedHighHz);
            const double adjacentHz = Overlap(band->fl, band->fh, adjacentLowHz, wantedLowHz) +
                                      Overlap(band->fl, band->fh, wantedHighHz, adjacentHighHz);
            m_totals.at(node).sourceAttributedWantedEnergyJ +=
                wattsPerHz * wantedHz * duration.GetSeconds();
            m_totals.at(node).sourceAttributedAdjacentEnergyJ +=
                wattsPerHz * adjacentHz * duration.GetSeconds();
        }
    }

    void Write() const
    {
        std::ofstream output(m_outputPath);
        NS_ABORT_MSG_IF(!output, "cannot open offline energy output: " << m_outputPath);
        output << "node,channel,sourceAttributedWantedEnergyJ,sourceAttributedAdjacentEnergyJ\n";
        for (uint32_t node = 0; node < m_totals.size(); ++node)
        {
            output << node << ','
                   << static_cast<unsigned>(m_devices.at(node)->GetPhy()->GetChannelNumber()) << ','
                   << m_totals.at(node).sourceAttributedWantedEnergyJ << ','
                   << m_totals.at(node).sourceAttributedAdjacentEnergyJ << '\n';
        }
    }

  private:
    struct Totals
    {
        double sourceAttributedWantedEnergyJ{0.0};
        double sourceAttributedAdjacentEnergyJ{0.0};
    };

    static double Overlap(double lowA, double highA, double lowB, double highB)
    {
        return std::max(0.0, std::min(highA, highB) - std::max(lowA, lowB));
    }

    std::vector<Totals> m_totals;
    std::vector<Ptr<WifiNetDevice>> m_devices;
    uint32_t m_emitterNodeId;
    std::string m_outputPath;
};

static OfflineEnergyRecorder* g_offlineEnergyRecorder = nullptr;

static void
OfflineSignalArrival(uint32_t node,
                     Ptr<const SpectrumSignalParameters> signal,
                     uint32_t senderNodeId,
                     double rxPower,
                     Time duration)
{
    g_offlineEnergyRecorder->SignalArrival(node, signal, senderNodeId, rxPower, duration);
}

// Runtime observation adapter: records receiver-local, unattributed PSD energy only.
#include "observed_energy_probe.h"
#include "ap_reception_audit.h"

static ObservedEnergyProbe* g_observedEnergyProbe = nullptr;
static void ApObservedSignalArrival(ObservedEnergyProbe* probe, uint32_t node,
                                    Ptr<const SpectrumValue> psd, Time duration)
{
    probe->SignalArrival(node, psd, duration);
}

static void
ObservedSignalArrival(uint32_t node, Ptr<const SpectrumValue> psd, Time duration)
{
    g_observedEnergyProbe->SignalArrival(node, psd, duration);
}

class TelemetryRiskController
{
  public:
    TelemetryRiskController(uint32_t numNodes,
                            double intervalSeconds,
                            double stopTime,
                            const std::string& outputPath,
                            bool enableGuard,
                            bool enableCapacityPacing,
                            bool initialRtsEnabled,
                            double retryThreshold,
                            double phyErrorThreshold,
                            double busyThreshold,
                            double backlogThreshold,
                            double globalBusyThreshold,
                            double globalBacklogThreshold,
                            double driftThreshold,
                            uint32_t minGuardDwellWindows,
                            uint32_t guardActivationWindows,
                            double collisionReleaseRatio,
                            double capacityReleaseRatio,
                            double pacingFloor)
        : m_windows(numNodes),
          m_busyHistory(numNodes),
          m_guardChangeHistory(numNodes),
          m_previousBusy(numNodes, 0.0),
          m_previousOffered(numNodes, 0.0),
          m_previousDelivery(numNodes, 0.0),
          m_previousSnr(numNodes, 0.0),
          m_hasPrevious(numNodes, false),
          m_executionController(
              numNodes,
              v33::HybridConfig{
                  std::max<uint32_t>(guardActivationWindows, 1),
                  minGuardDwellWindows,
                  Clamp01(collisionReleaseRatio),
                  Clamp01(capacityReleaseRatio),
                  pacingFloor,
              },
              enableGuard && enableCapacityPacing
                  ? v33::HybridExecutionMode::Hybrid
              : enableGuard
                  ? v33::HybridExecutionMode::TelemetryRts
              : enableCapacityPacing
                  ? v33::HybridExecutionMode::CapacityPacing
                  : v33::HybridExecutionMode::Baseline,
              initialRtsEnabled),
          m_intervalSeconds(intervalSeconds),
          m_stopTime(stopTime),
          m_lastSampleTime(1.0),
          m_retryThreshold(std::max(retryThreshold, 1e-6)),
          m_phyErrorThreshold(std::max(phyErrorThreshold, 1e-6)),
          m_busyThreshold(std::max(busyThreshold, 1e-6)),
          m_backlogThreshold(std::max(backlogThreshold, 1e-6)),
          m_globalBusyThreshold(std::max(globalBusyThreshold, 1e-6)),
          m_globalBacklogThreshold(std::max(globalBacklogThreshold, 1e-6)),
          m_driftThreshold(std::max(driftThreshold, 1e-6))
    {
        m_output.open(outputPath);
        NS_ABORT_MSG_IF(!m_output, "cannot open telemetry output: " << outputPath);
        v33::ConfigureTelemetryStream(m_output);
        m_output << "time,node,appTxPackets,appRxPackets,offeredMbps,deliveredMbps,"
                    "retryEvents,finalFailures,retryRate,phyErrorRate,busyRatio,"
                    "backlogRatio,busyVariance,snrMean,snrVariance,offeredChange,"
                    "deliveryChange,snrChange,globalBusyRatio,globalBacklogRatio,"
                    "zCollision,zCapacity,zDrift,zOscillation,"
                    "collisionActive,capacityActive,driftActive,oscillationActive,"
                    "rtsEnabled,guardChanged,riskClass,nominalDemandMbps,"
                    "admittedMbps,deferredDemandBytes,demandSatisfaction,"
                    "pacingLevel,pacingChanged,macTxPackets,phyRxOk,phyRxError,appTxBytes,appRxBytes,queuePackets,queueBytes,currentChannel,channelChanged,channelSwitchCount,adjacentObservationAvailable,"
                    "windowElapsedSeconds,nominalDemandBytes,admittedDemandBytes,deliveredBytes\n";
    }

    void AddDevice(Ptr<WifiNetDevice> device)
    {
        m_devices.push_back(device);
        m_previousChannel.push_back(device->GetPhy()->GetChannelNumber());
        m_channelSwitchCount.push_back(0);
    }

    void AddManager(Ptr<WifiRemoteStationManager> manager)
    {
        m_managers.push_back(manager);
    }

    void AddApplication(Ptr<OnOffApplication> app, DataRate nominalRate,uint64_t burstBaseBps=0)
    {
        NS_ABORT_MSG_IF(!app, "v33 pacing requires OnOffApplication");
        m_sources.push_back(app);
        m_nominalRates.push_back(burstBaseBps?DataRate(burstBaseBps):nominalRate);
        m_admittedRates.push_back(nominalRate);
        m_fixedBurstDemand.push_back(burstBaseBps);
        m_nominalOnState.push_back(false);
        m_admittedOnState.push_back(false);
        m_channelPaused.push_back(false);
        m_demandLastUpdate.push_back(Simulator::Now().GetSeconds());
        m_nominalDemandBytes.push_back(0.0);
        m_admittedDemandBytes.push_back(0.0);
    }

    void StopDemand()
    {
        for (uint32_t node = 0; node < m_nominalOnState.size(); ++node)
        {
            AdvanceDemandLedger(node, Simulator::Now().GetSeconds());
            m_nominalOnState[node] = false;
            m_admittedOnState[node] = false;
        }
    }

    void Finalize()
    {
        if (Simulator::Now().GetSeconds() > m_lastSampleTime)
        {
            Sample();
        }
    }

    void SetProspectiveRate(uint32_t node,uint64_t bps)
    {
        NS_ABORT_MSG_IF(!bps,"positive prospective offered rate required");
        AdvanceDemandLedger(node,Simulator::Now().GetSeconds());
        m_nominalRates.at(node)=DataRate(bps);m_admittedRates.at(node)=DataRate(bps);
        // Public OnOff DataRate semantics retain the current send event/cached
        // packet; no stop/start, CancelEvents or Wi-Fi queue operation occurs.
        m_sources.at(node)->SetAttribute("DataRate",DataRateValue(DataRate(bps)));
    }

    void OnOffState(uint32_t node, bool onState)
    {
        AdvanceDemandLedger(node, Simulator::Now().GetSeconds());
        m_nominalOnState.at(node) = onState;
        m_admittedOnState.at(node) = onState && !m_channelPaused.at(node);
    }

    void BeginChannelPause(uint32_t node)
    {
        AdvanceDemandLedger(node, Simulator::Now().GetSeconds());
        NS_ABORT_MSG_IF(m_channelPaused.at(node), "duplicate channel pause");
        m_channelPaused.at(node) = true;
        // Nominal demand remains on as a counterfactual ledger; admitted
        // generation is explicitly paused until the actuation resolves.
        m_admittedOnState.at(node) = false;
    }

    void EndChannelPause(uint32_t node)
    {
        AdvanceDemandLedger(node, Simulator::Now().GetSeconds());
        NS_ABORT_MSG_IF(!m_channelPaused.at(node), "channel pause not active");
        m_channelPaused.at(node) = false;
        m_admittedOnState.at(node) = m_nominalOnState.at(node);
    }

    void AppTx(uint32_t node, Ptr<const Packet> packet)
    {
        ++m_windows.at(node).appTxPackets;
        m_windows.at(node).appTxBytes += packet->GetSize();
    }

    void AppRx(uint32_t node, Ptr<const Packet> packet)
    {
        ++m_windows.at(node).appRxPackets;
        m_windows.at(node).appRxBytes += packet->GetSize();
    }

    void MacTx(uint32_t node)
    {
        ++m_windows.at(node).macTxPackets;
    }

    void Retry(uint32_t node)
    {
        ++m_windows.at(node).retryEvents;
    }

    void FinalFailure(uint32_t node)
    {
        ++m_windows.at(node).finalFailures;
    }

    void PhyRxOk(uint32_t node, double snr)
    {
        ++m_windows.at(node).phyRxOk;
        m_windows.at(node).snrSum += snr;
        m_windows.at(node).snrSquaredSum += snr * snr;
    }

    void PhyRxError(uint32_t node)
    {
        ++m_windows.at(node).phyRxError;
    }

    void PhyState(uint32_t node, Time start, Time duration, WifiPhyState state)
    {
        if (state == WifiPhyState::CCA_BUSY || state == WifiPhyState::TX ||
            state == WifiPhyState::RX)
        {
            // PHY state intervals are short relative to the telemetry window. Attribute an
            // interval to the window in which its trace fires; Clamp01 handles boundary jitter.
            m_windows.at(node).busySeconds += duration.GetSeconds();
        }
    }

    void Sample()
    {
        NS_ABORT_MSG_IF(m_managers.size() != m_windows.size(),
                        "v33 manager count must equal Node count");
        NS_ABORT_MSG_IF(m_sources.size() != m_windows.size(),
                        "v33 application count must equal Node count");
        NS_ABORT_MSG_IF(m_nominalRates.size() != m_windows.size(),
                        "v33 nominal-rate count must equal Node count");

        const double now = Simulator::Now().GetSeconds();
        const double elapsed = now - m_lastSampleTime;
        NS_ABORT_MSG_IF(elapsed <= 0.0, "telemetry sample elapsed time must be positive");
        for (uint32_t node = 0; node < m_windows.size(); ++node)
        {
            AdvanceDemandLedger(node, now);
        }
        uint64_t globalTxPackets = 0;
        uint64_t globalRxPackets = 0;
        double globalBusySeconds = 0.0;
        for (const auto& window : m_windows)
        {
            globalTxPackets += window.appTxPackets;
            globalRxPackets += window.appRxPackets;
            globalBusySeconds += window.busySeconds;
        }
        const double globalBusyRatio = Clamp01(
            globalBusySeconds / (elapsed * std::max<std::size_t>(m_windows.size(), 1)));
        const double globalBacklogRatio = globalTxPackets > 0
                                              ? Clamp01(
                                                    1.0 - static_cast<double>(globalRxPackets) /
                                                              globalTxPackets)
                                              : 0.0;
        const bool globalCapacityActive = globalBusyRatio >= m_globalBusyThreshold &&
                                          globalBacklogRatio >= m_globalBacklogThreshold;
        std::vector<double> retryRates(m_windows.size(), 0.0);
        std::vector<double> phyErrorRates(m_windows.size(), 0.0);
        std::vector<double> busyRatios(m_windows.size(), 0.0);
        std::vector<double> backlogRatios(m_windows.size(), 0.0);
        std::vector<double> zCollisions(m_windows.size(), 0.0);
        std::vector<double> zCapacities(m_windows.size(), 0.0);
        std::vector<bool> collisionActiveByNode(m_windows.size(), false);
        std::vector<bool> localCapacityActiveByNode(m_windows.size(), false);
        std::vector<v33::HybridInput> inputs;
        inputs.reserve(m_windows.size());
        for (uint32_t node = 0; node < m_windows.size(); ++node)
        {
            const auto& window = m_windows[node];
            retryRates[node] = static_cast<double>(window.retryEvents) /
                               std::max<uint64_t>(window.macTxPackets, 1);
            phyErrorRates[node] = static_cast<double>(window.phyRxError) /
                                  std::max<uint64_t>(window.phyRxOk + window.phyRxError, 1);
            busyRatios[node] = Clamp01(window.busySeconds / elapsed);
            backlogRatios[node] = window.appTxPackets > 0
                                      ? Clamp01(1.0 - static_cast<double>(window.appRxPackets) /
                                                            window.appTxPackets)
                                      : 0.0;
            zCollisions[node] = Clamp01(std::max(retryRates[node] / m_retryThreshold,
                                                  phyErrorRates[node] / m_phyErrorThreshold));
            zCapacities[node] = Clamp01(std::sqrt(
                Clamp01(busyRatios[node] / m_busyThreshold) *
                Clamp01(backlogRatios[node] / m_backlogThreshold)));
            collisionActiveByNode[node] = retryRates[node] >= m_retryThreshold ||
                                          phyErrorRates[node] >= m_phyErrorThreshold;
            localCapacityActiveByNode[node] = busyRatios[node] >= m_busyThreshold &&
                                              backlogRatios[node] >= m_backlogThreshold;
            inputs.push_back({collisionActiveByNode[node],
                              localCapacityActiveByNode[node],
                              globalCapacityActive,
                              zCollisions[node],
                              zCapacities[node]});
        }
        const auto executions = m_executionController.UpdateWindow(inputs);

        for (uint32_t node = 0; node < m_windows.size(); ++node)
        {
            const auto& window = m_windows[node];
            const double offeredMbps = window.appTxBytes * 8.0 / elapsed / 1e6;
            const double deliveredMbps = window.appRxBytes * 8.0 / elapsed / 1e6;
            const double retryRate = retryRates[node];
            const double phyErrorRate = phyErrorRates[node];
            const double busyRatio = busyRatios[node];
            const double backlogRatio = backlogRatios[node];

            auto& history = m_busyHistory[node];
            history.push_back(busyRatio);
            if (history.size() > 4)
            {
                history.pop_front();
            }
            const double busyMean = std::accumulate(history.begin(), history.end(), 0.0) /
                                    std::max<std::size_t>(history.size(), 1);
            double busyVariance = 0.0;
            for (double value : history)
            {
                busyVariance += (value - busyMean) * (value - busyMean);
            }
            busyVariance /= std::max<std::size_t>(history.size(), 1);
            const double snrMean = window.phyRxOk > 0
                                       ? window.snrSum / window.phyRxOk
                                       : 0.0;
            const double snrVariance = window.phyRxOk > 0
                                           ? std::max(0.0,
                                                      window.snrSquaredSum / window.phyRxOk -
                                                          snrMean * snrMean)
                                           : 0.0;

            const double zCollision = zCollisions[node];
            const double zCapacity = zCapacities[node];
            double offeredChange = 0.0;
            double deliveryChange = 0.0;
            double busyChange = 0.0;
            double snrChange = 0.0;
            if (m_hasPrevious[node])
            {
                offeredChange =
                    std::abs(offeredMbps - m_previousOffered[node]) /
                    std::max(std::max(offeredMbps, m_previousOffered[node]), 1.0);
                deliveryChange =
                    std::abs(deliveredMbps - m_previousDelivery[node]) /
                    std::max(std::max(offeredMbps, m_previousOffered[node]), 1.0);
                busyChange = std::abs(busyRatio - m_previousBusy[node]);
                snrChange = std::abs(snrMean - m_previousSnr[node]) /
                            std::max(std::abs(m_previousSnr[node]), 1.0);
            }
            const double pressureVariation = std::sqrt(std::max(0.0, busyVariance));
            const double zDrift = Clamp01(std::max(
                {offeredChange, deliveryChange, busyChange, snrChange, pressureVariation}) /
                                          m_driftThreshold);
            const auto& guardHistory = m_guardChangeHistory[node];
            const double zOscillation = guardHistory.empty()
                                            ? 0.0
                                            : std::accumulate(guardHistory.begin(),
                                                              guardHistory.end(),
                                                              0.0) /
                                                  guardHistory.size();

            const bool collisionActive = collisionActiveByNode[node];
            const bool localCapacityActive = localCapacityActiveByNode[node];
            const bool capacityActive = globalCapacityActive || localCapacityActive;
            const bool driftActive = zDrift >= 1.0;
            const bool oscillationActive = zOscillation >= 0.5;
            const auto& execution = executions[node];
            if (execution.guardChanged)
            {
                m_managers[node]->SetAttribute("RtsCtsThreshold",
                                               UintegerValue(execution.rtsEnabled ? 0 : 65535));
            }
            auto& changeHistory = m_guardChangeHistory[node];
            changeHistory.push_back(execution.guardChanged ? 1.0 : 0.0);
            if (changeHistory.size() > 4)
            {
                changeHistory.pop_front();
            }

            const uint64_t nominalBps = m_nominalRates[node].GetBitRate();
            const uint64_t sourceBps = m_fixedBurstDemand[node]?BurstPeakBps(m_fixedBurstDemand[node]):nominalBps;
            const uint64_t elapsedPacedBps = static_cast<uint64_t>(
                std::llround(sourceBps * execution.elapsedPacingLevel));
            const uint64_t nextPacedBps = static_cast<uint64_t>(
                std::llround(sourceBps * execution.nextPacingLevel));
            m_sources[node]->SetAttribute("DataRate", DataRateValue(DataRate(nextPacedBps)));
            m_admittedRates[node] = DataRate(nextPacedBps);

            const double nominalBytes = m_nominalDemandBytes.at(node);
            const double admittedBytes = m_admittedDemandBytes.at(node);
            const double deferredBytes = std::max(0.0, nominalBytes - admittedBytes);
            const double demandSatisfaction =
                nominalBytes > 0.0
                    ? std::min(1.0,
                               static_cast<double>(window.appRxBytes) / nominalBytes)
                    : 1.0;

            // Snapshot of actual STA transmit queues. Sampled channel changes do not
            // detect multiple switches within a window; this probe does not switch channels.
            uint32_t queuePackets = 0;
            uint32_t queueBytes = 0;
            auto mac = m_devices.at(node)->GetMac();
            if (mac->GetQosSupported())
            {
                for (auto ac : {AC_BE, AC_BK, AC_VI, AC_VO})
                {
                    auto queue = mac->GetTxopQueue(ac);
                    if (queue) { queuePackets += queue->GetNPackets(); queueBytes += queue->GetNBytes(); }
                }
            }
            else
            {
                auto queue = mac->GetTxop()->GetWifiMacQueue();
                queuePackets = queue->GetNPackets();
                queueBytes = queue->GetNBytes();
            }
            const auto channel = m_devices.at(node)->GetPhy()->GetChannelNumber();
            const bool channelChanged = channel != m_previousChannel.at(node);
            m_channelSwitchCount.at(node) += channelChanged;
            m_previousChannel.at(node) = channel;

            m_output << now << ',' << node << ',' << window.appTxPackets << ','
                     << window.appRxPackets << ',' << offeredMbps << ',' << deliveredMbps << ','
                     << window.retryEvents << ',' << window.finalFailures << ',' << retryRate << ','
                     << phyErrorRate << ',' << busyRatio << ',' << backlogRatio << ','
                     << busyVariance << ',' << snrMean << ',' << snrVariance << ','
                     << offeredChange << ',' << deliveryChange << ',' << snrChange << ','
                     << globalBusyRatio << ',' << globalBacklogRatio << ','
                     << zCollision << ',' << zCapacity << ',' << zDrift << ',' << zOscillation << ','
                     << collisionActive << ',' << capacityActive << ','
                     << driftActive << ',' << oscillationActive << ',' << execution.rtsEnabled << ','
                     << execution.guardChanged << ',' << RiskClassName(execution.riskClass) << ','
                     << nominalBps / 1e6 << ',' << elapsedPacedBps / 1e6 << ',' << deferredBytes
                     << ',' << demandSatisfaction << ',' << execution.elapsedPacingLevel << ','
                     << execution.pacingChanged << ',' << window.macTxPackets << ','
                     << window.phyRxOk << ',' << window.phyRxError << ','
                     << window.appTxBytes << ',' << window.appRxBytes << ','
                     << queuePackets << ',' << queueBytes << ',' << static_cast<unsigned>(channel) << ','
                     << channelChanged << ',' << m_channelSwitchCount.at(node) << ','
                     << 0 << ',' << elapsed << ',' << nominalBytes << ',' << admittedBytes << ','
                     << static_cast<double>(window.appRxBytes) << '\n';

            m_previousBusy[node] = busyRatio;
            m_previousOffered[node] = offeredMbps;
            m_previousDelivery[node] = deliveredMbps;
            m_previousSnr[node] = snrMean;
            m_hasPrevious[node] = true;
            m_nominalDemandBytes[node] = 0.0;
            m_admittedDemandBytes[node] = 0.0;
        }
        m_output.flush();
        std::fill(m_windows.begin(), m_windows.end(), TelemetryWindow{});
        m_lastSampleTime = now;
        if (now + m_intervalSeconds <= m_stopTime + 1e-9)
        {
            Simulator::Schedule(Seconds(m_intervalSeconds),
                                &TelemetryRiskController::Sample,
                                this);
        }
    }

  private:
    void AdvanceDemandLedger(uint32_t node, double now)
    {
        const double elapsed = now - m_demandLastUpdate.at(node);
        NS_ABORT_MSG_IF(elapsed < 0.0, "demand ledger time must not move backward");
        if(m_fixedBurstDemand.at(node))
        {
            m_nominalDemandBytes.at(node)+=BurstNominalBytes(m_fixedBurstDemand[node],
                Seconds(m_demandLastUpdate.at(node)).GetNanoSeconds(),Seconds(now).GetNanoSeconds());
        }
        else if (m_nominalOnState.at(node))
        {
            m_nominalDemandBytes.at(node) +=
                m_nominalRates.at(node).GetBitRate() * elapsed / 8.0;
        }
        if (m_admittedOnState.at(node))
        {
            m_admittedDemandBytes.at(node) +=
                m_admittedRates.at(node).GetBitRate() * elapsed / 8.0;
        }
        m_demandLastUpdate.at(node) = now;
    }

    std::vector<TelemetryWindow> m_windows;
    std::vector<std::deque<double>> m_busyHistory;
    std::vector<std::deque<double>> m_guardChangeHistory;
    std::vector<double> m_previousBusy;
    std::vector<double> m_previousOffered;
    std::vector<double> m_previousDelivery;
    std::vector<double> m_previousSnr;
    std::vector<bool> m_hasPrevious;
    std::vector<Ptr<WifiNetDevice>> m_devices;
    std::vector<uint16_t> m_previousChannel;
    std::vector<uint64_t> m_channelSwitchCount;
    std::vector<Ptr<WifiRemoteStationManager>> m_managers;
    std::vector<Ptr<OnOffApplication>> m_sources;
    std::vector<DataRate> m_nominalRates;
    std::vector<uint64_t> m_fixedBurstDemand;
    std::vector<DataRate> m_admittedRates;
    std::vector<bool> m_nominalOnState;
    std::vector<bool> m_admittedOnState;
    std::vector<bool> m_channelPaused;
    std::vector<double> m_demandLastUpdate;
    std::vector<double> m_nominalDemandBytes;
    std::vector<double> m_admittedDemandBytes;
    v33::HybridExecutionController m_executionController;
    double m_intervalSeconds;
    double m_stopTime;
    double m_lastSampleTime;
    double m_retryThreshold;
    double m_phyErrorThreshold;
    double m_busyThreshold;
    double m_backlogThreshold;
    double m_globalBusyThreshold;
    double m_globalBacklogThreshold;
    double m_driftThreshold;
    std::ofstream m_output;
};

static TelemetryRiskController* g_telemetryController = nullptr;
static PacketCohort* g_packetCohort = nullptr;

static void
TelemetryAppTx(uint32_t node, Ptr<const Packet> packet)
{
    g_telemetryController->AppTx(node, packet);
    g_corrected->App(node, packet, false);
    if(g_packetCohort)g_packetCohort->Tx(node,packet);
}

static void
TelemetryAppRx(uint32_t node, Ptr<const Packet> packet, const Address&)
{
    g_telemetryController->AppRx(node, packet);
    g_corrected->App(node, packet, true);
    if(g_packetCohort)g_packetCohort->Rx(node,packet);
}

static void
TelemetryOnOffState(uint32_t node, bool, bool onState)
{
    g_telemetryController->OnOffState(node, onState);
}

static void
TelemetryMacTx(uint32_t node, Ptr<const Packet>)
{
    g_telemetryController->MacTx(node);
}

static void
TelemetryRetry(uint32_t node, Mac48Address)
{
    g_telemetryController->Retry(node);
}

static void
TelemetryFinalFailure(uint32_t node, Mac48Address)
{
    g_telemetryController->FinalFailure(node);
}

static void
TelemetryPhyRxOk(uint32_t node,
                 Ptr<const Packet>,
                 double snr,
                 WifiMode,
                 WifiPreamble)
{
    g_telemetryController->PhyRxOk(node, snr);
    g_corrected->Rx(node, snr);
}

static void
TelemetryPhyRxError(uint32_t node, Ptr<const Packet>, double)
{
    g_telemetryController->PhyRxError(node);
}

static void
TelemetryPhyState(uint32_t node, Time start, Time duration, WifiPhyState state)
{
    g_telemetryController->PhyState(node, start, duration, state);
}

static std::vector<uint32_t>
ParseAssignment(const std::string& text, uint32_t numNodes, uint32_t numChannels)
{
    std::vector<uint32_t> assignment;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ','))
    {
        if (!token.empty())
        {
            assignment.push_back(static_cast<uint32_t>(std::stoul(token)));
        }
    }
    if (assignment.size() != numNodes)
    {
        NS_FATAL_ERROR("assignment must contain exactly " << numNodes << " values");
    }
    for (auto channel : assignment)
    {
        if (channel >= numChannels)
        {
            NS_FATAL_ERROR("assignment contains a channel outside [0, " << numChannels - 1
                           << "]");
        }
    }
    return assignment;
}

static std::string
ChannelSettings(uint32_t channelIndex)
{
    // Four non-overlapping 20 MHz 5 GHz channels for the first validation pass.
    const uint32_t channelNumber = 36 + 4 * channelIndex;
    return "{" + std::to_string(channelNumber) + ", 20, BAND_5GHZ, 0}";
}

struct ChannelQuiescenceListener : WifiPhyListener
{
    int64_t lastActivityEnd=0,lastFrameEnd=0;
    const bool* measuring=nullptr;
    void Activity(Time duration=Seconds(0))
    {
        lastActivityEnd=std::max(lastActivityEnd,
            (Simulator::Now()+duration).GetNanoSeconds());
        lastFrameEnd=std::max(lastFrameEnd,(Simulator::Now()+duration).GetNanoSeconds());
    }
    void NotifyRxStart(Time d) override {Activity(d);}
    void NotifyRxEndOk() override {Activity();}
    void NotifyRxEndError(const WifiTxVector&) override {Activity();}
    void NotifyTxStart(Time d,dBm_u) override {
        NS_ABORT_MSG_IF(measuring&&*measuring,"data/management TX during held measurement scan");Activity(d);
    }
    void NotifyCcaBusyStart(Time d,WifiChannelListType,const std::vector<Time>&) override {
        lastActivityEnd=std::max(lastActivityEnd,(Simulator::Now()+d).GetNanoSeconds());
    }
    void NotifySwitchingStart(Time d) override {Activity(d);}
    void NotifySleep() override {Activity();}
    void NotifyOff() override {Activity();}
    void NotifyWakeup() override {Activity();}
    void NotifyOn() override {Activity();}
};

class ChannelController
{
  public:
    ChannelController(std::vector<Ptr<WifiNetDevice>> aps,
                      std::vector<Ptr<WifiNetDevice>> stas,
                      NetDeviceContainer allWifi,
                      Ptr<WaveformGenerator> waveform,
                       Ptr<ConstantSpeedPropagationDelayModel> delay,
                       CorrectedTelemetry& telemetry,
                       TelemetryRiskController& demandLedger,
                       std::vector<Ptr<OnOffApplication>> sources,
                       double interval,
                      double stop,
                      std::string schedule,
                      std::string output,
                      QueueContinuity* queueContinuity=nullptr,bool windowTracking=false)
        : m_aps(std::move(aps)), m_stas(std::move(stas)), m_allWifi(std::move(allWifi)),
           m_waveform(std::move(waveform)), m_delay(std::move(delay)), m_telemetry(telemetry),
           m_demandLedger(demandLedger), m_sources(std::move(sources)), m_interval(interval),
           m_stop(stop), m_schedule(std::move(schedule)), m_output(output), m_scanDiagnostic(output+".scan-detail.csv"),
           m_scanRecovery(output+".scan-recovery.csv"),m_pending(m_stas.size()),
           m_apActivity(m_stas.size()),m_staActivity(m_stas.size()),m_done(m_stas.size()),m_queueContinuity(queueContinuity),m_windowTracking(windowTracking),m_tracking(output+".tracking.csv")
    {
        NS_ABORT_MSG_IF(!m_tracking,"tracking audit unavailable");
        m_tracking<<"epoch,timeNs,event,node,role,activeChannel,registeredFlags,ready\n";
        NS_ABORT_MSG_IF(m_aps.size()!=15 || m_stas.size()!=15, "channel controller requires 15 pairs");
        NS_ABORT_MSG_IF(m_sources.size()!=m_stas.size(), "channel controller requires one source per pair");
        NS_ABORT_MSG_IF(m_schedule!="quiet" && m_schedule!="two_window" && m_schedule!="isolated" && m_schedule!="external",
                        "unknown channel control schedule");
        NS_ABORT_MSG_IF(!m_output, "cannot open channel control trace");
        NS_ABORT_MSG_IF(!m_scanDiagnostic,"cannot open scan diagnostic");
        m_scanDiagnostic<<"epoch,timeNs,node,apState,staState,associated,channel,apLastActivityEndNs,staLastActivityEndNs,responseIdle,guardNs\n";
        NS_ABORT_MSG_IF(!m_scanRecovery,"cannot open scan recovery trace");
        m_scanRecovery<<"epoch,node,restoredAtNs,releasedAtNs,apState,staState,associated,channel\n";
        m_output << "timeS,node,opportunity,oldChannel,newChannel,changed,status,"
                    "pauseDurationNs,queuesEmpty,physIdle,responseTimersIdle,quietGuardNs,schedule\n";
        for(uint32_t node=0;node<m_stas.size();++node)
        {
            m_apActivity[node]=std::make_shared<ChannelQuiescenceListener>();
            m_staActivity[node]=std::make_shared<ChannelQuiescenceListener>();
            m_apActivity[node]->measuring=&m_measuring;m_staActivity[node]->measuring=&m_measuring;
            m_aps[node]->GetPhy()->RegisterListener(m_apActivity[node]);
            m_stas[node]->GetPhy()->RegisterListener(m_staActivity[node]);
        }
    }
    void Start() { if(m_windowTracking)Simulator::Schedule(Seconds(.4),[this]{CloseTracking(0);}); if(m_schedule!="external")Simulator::Schedule(Seconds(1.0), &ChannelController::Tick, this); }
    uint8_t ReadChannel(uint32_t node) const {
        const auto current=m_stas.at(node)->GetPhy()->GetChannelNumber();
        NS_ABORT_MSG_IF(current!=m_aps.at(node)->GetPhy()->GetChannelNumber()||current<36||current>48||(current-36)%4,
                        "invalid paired channel readback");return (current-36)/4;
    }

    void Execute(uint32_t node,uint8_t target,uint64_t epoch,threeway::ManagementEpochs::Done done) {
        NS_ABORT_MSG_IF(m_schedule!="external"||node>=m_stas.size()||target>=4,"invalid external channel request");
        if(m_pending[node].active){done(threeway::ManagementHeader::BUSY);return;}
        NS_ABORT_MSG_IF(m_fresh&&(!TrackingWindowReady(m_aps[node]->GetPhy())||!TrackingWindowReady(m_stas[node]->GetPhy())),"permanent action before tracking ready");
        m_telemetry.ControlOpportunity(node);
        const auto old=m_stas[node]->GetPhy()->GetChannelNumber();
        if(ReadChannel(node)==target) {
            Trace(node,epoch,old,old,false,"unchanged",0,true,true,true);done(threeway::ManagementHeader::OK);return;
        }
        m_done[node]=std::move(done);Begin(node,epoch,old,36+4*target);
    }
    void SetFreshSounding(FreshSounding* fresh) {m_fresh=fresh;}
    void TraceTracking(uint64_t epoch,const char* event,bool requireReady=false) {
        for(unsigned n=0;n<m_stas.size();++n)for(unsigned role=0;role<2;++role) {
            auto phy=DynamicCast<SpectrumWifiPhy>((role?m_stas[n]:m_aps[n])->GetPhy());
            std::string flags;bool activeRegistered=false;
            for(auto& [range,iface]:phy->GetSpectrumPhyInterfaces()) {
                auto channel=DynamicCast<AcquisitionSpectrumChannel>(iface->GetChannel());
                bool registered=channel&&channel->HasHistory(iface,NanoSeconds(0));flags+=registered?'1':'0';
                if(iface->GetCenterFrequencies()==phy->GetOperatingChannel().GetFrequencies())activeRegistered=registered;
            }
            const bool ready=TrackingWindowReady(phy);
            NS_ABORT_MSG_IF(!activeRegistered||(requireReady&&!ready),"tracking certificate/active continuity invalid");
            m_tracking<<epoch<<','<<Simulator::Now().GetNanoSeconds()<<','<<event<<','<<n<<','<<(role?"STA":"AP")<<','
                <<unsigned(phy->GetChannelNumber())<<','<<flags<<','<<ready<<'\n';
        }
        m_tracking.flush();
    }
    void OpenTracking(uint64_t epoch) {
        m_trackingEpoch=epoch;
        if(m_windowTracking)for(unsigned i=0;i<m_allWifi.GetN();++i) {
            auto device=m_allWifi.Get(i);
            auto phy=DynamicCast<SpectrumWifiPhy>(DynamicCast<WifiNetDevice>(device)->GetPhy());
            for(auto& [range,iface]:phy->GetSpectrumPhyInterfaces()) {
                auto channel=DynamicCast<AcquisitionSpectrumChannel>(iface->GetChannel());
                NS_ABORT_MSG_IF(!channel,"unobserved spectrum channel");
                if(!channel->HasHistory(iface,NanoSeconds(0)))channel->AddRx(iface);
            }
        }
        TraceTracking(epoch,"warm_open");
    }
    void CloseTracking(uint64_t epoch) {
        if(!m_windowTracking||epoch!=m_trackingEpoch)return;
        if(m_scanInProgress||std::any_of(m_pending.begin(),m_pending.end(),[](const Pending& p){return p.active;})) {
            Simulator::Schedule(MicroSeconds(100),[this,epoch]{CloseTracking(epoch);});return;
        }
        TraceTracking(epoch,"before_close",true);
        for(unsigned i=0;i<m_allWifi.GetN();++i) {
            auto device=m_allWifi.Get(i);
            auto phy=DynamicCast<SpectrumWifiPhy>(DynamicCast<WifiNetDevice>(device)->GetPhy());
            for(auto& [range,iface]:phy->GetSpectrumPhyInterfaces())
                if(iface->GetCenterFrequencies()!=phy->GetOperatingChannel().GetFrequencies())iface->GetChannel()->RemoveRx(iface);
        }
        TraceTracking(epoch,"closed_active_only");
    }
    // Acquisition diagnostic only; policy decisions remain disabled.
    void ProbeScanQuiescence(uint64_t epoch,std::ostream& output) {
        NS_ABORT_MSG_IF(m_schedule!="quiet"&&!(m_schedule=="external"&&m_fresh),"scan requires quiet or fresh external control");
        NS_ABORT_MSG_IF(m_scanInProgress,"overlapping acquisition scan");m_scanInProgress=true;TraceTracking(epoch,"scan_gate",true);
        const auto started=Simulator::Now().GetNanoSeconds();
        for(unsigned n=0;n<m_sources.size();++n) {
            NS_ABORT_MSG_IF(m_pending[n].active,"scan conflicts with pending channel control");
            m_aps[n]->GetMac()->SetAttribute("BeaconGeneration",BooleanValue(false));
            m_aps[n]->GetMac()->GetChannelAccessManager()->SetMeasurementHold(true);
            m_stas[n]->GetMac()->GetChannelAccessManager()->SetMeasurementHold(true);
            QueueCheckpoint(n,epoch,"before_pause");m_demandLedger.BeginChannelPause(n);
            NS_ABORT_MSG_IF(!m_sources[n]->PauseForChannelSwitch(),"scan source refused pause");
            QueueCheckpoint(n,epoch,"after_pause");
        }
        Simulator::Schedule(MicroSeconds(50),[this,epoch,started,&output]{PollScanQuiescence(epoch,started,output);});
    }
  private:
    void PollScanQuiescence(uint64_t epoch,int64_t started,std::ostream& output) {
        const auto now=Simulator::Now().GetNanoSeconds();
        unsigned nonempty=0,busy=0,response=0,guard=0;
        for(unsigned n=0;n<m_sources.size();++n) {
            nonempty+=!(QueueEmpty(m_aps[n])&&QueueEmpty(m_stas[n]));
            busy+=!(ScanRetunable(m_aps[n]->GetPhy()->GetState()->GetState())&&ScanRetunable(m_stas[n]->GetPhy()->GetState()->GetState()));
            response+=!(ResponseTimerIdle(m_aps[n])&&ResponseTimerIdle(m_stas[n]));
            guard+=now-std::max(m_apActivity[n]->lastFrameEnd,m_staActivity[n]->lastFrameEnd)<ResponseGuard(n).GetNanoSeconds();
        }
        const auto preparation=MilliSeconds(m_fresh?37:182).GetNanoSeconds();
        const bool ready=!busy&&!response&&!guard&&now-started<=preparation;
        if(!ready && now-started<preparation) {
            Simulator::Schedule(MicroSeconds(50),[this,epoch,started,&output]{PollScanQuiescence(epoch,started,output);});return;
        }
        for(unsigned n=0;n<m_sources.size();++n) {
            m_scanDiagnostic<<epoch<<','<<now<<','<<n<<','<<m_aps[n]->GetPhy()->GetState()->GetState()<<','
                <<m_stas[n]->GetPhy()->GetState()->GetState()<<','<<DynamicCast<StaWifiMac>(m_stas[n]->GetMac())->IsAssociated()<<','
                <<unsigned(m_stas[n]->GetPhy()->GetChannelNumber())<<','<<m_apActivity[n]->lastActivityEnd<<','
                <<m_staActivity[n]->lastActivityEnd<<','<<(ResponseTimerIdle(m_aps[n])&&ResponseTimerIdle(m_stas[n]))<<','
                <<ResponseGuard(n).GetNanoSeconds()<<'\n';
        }
        if(ready) {
            m_measuring=true;std::vector<uint8_t> original;
            for(unsigned n=0;n<m_sources.size();++n)original.push_back(ReadChannel(n));
            if(m_fresh)m_fresh->Begin(epoch);
            for(unsigned c=0;c<4;++c)Simulator::Schedule(MicroSeconds(1000+(m_fresh?m_fresh->Schedule().CycleUs():4000)*c),[this,c,epoch]{
                ScanRetune(epoch*10+c,std::vector<uint8_t>(m_sources.size(),c));
            });
            Simulator::Schedule(MicroSeconds(m_fresh?m_fresh->Schedule().RestoreUs():17000),[this,epoch,original]{ScanRetune(epoch*10+4,original);});
            Simulator::Schedule(MicroSeconds(m_fresh?m_fresh->Schedule().FinishUs():18000),[this,epoch,started,nonempty,original,&output]{
                NS_ABORT_MSG_IF(Simulator::Now().GetNanoSeconds()-started>200000000,"total scan budget exceeded");
                for(unsigned n=0;n<m_sources.size();++n)
                    NS_ABORT_MSG_IF(!ScanRetunable(m_aps[n]->GetPhy()->GetState()->GetState())||!ScanRetunable(m_stas[n]->GetPhy()->GetState()->GetState())||
                        !ResponseTimerIdle(m_aps[n])||!ResponseTimerIdle(m_stas[n])||ReadChannel(n)!=original[n],"scan restoration not safe");
                if(m_fresh)m_fresh->Finish();
                const auto restored=Simulator::Now().GetNanoSeconds();
                bool retained=true;
                for(unsigned n=0;n<m_sources.size();++n)
                    retained=retained&&TrackingWindowReady(m_aps[n]->GetPhy())&&TrackingWindowReady(m_stas[n]->GetPhy());
                // Configured interfaces retain ongoing interference; legacy interfaces still need recovery.
                Simulator::Schedule(retained?NanoSeconds(0):RecoveryCoverageBound(),[this,epoch,started,nonempty,restored,original,&output]{
                    NS_ABORT_MSG_IF(Simulator::Now().GetNanoSeconds()-started>200000000,"scan and recovery exceed hold budget");
                    for(unsigned n=0;n<m_sources.size();++n) {
                        NS_ABORT_MSG_IF(!ScanRetunable(m_aps[n]->GetPhy()->GetState()->GetState())||!ScanRetunable(m_stas[n]->GetPhy()->GetState()->GetState())||
                            !ResponseTimerIdle(m_aps[n])||!ResponseTimerIdle(m_stas[n])||ReadChannel(n)!=original[n],"scan release not safe");
                        m_scanRecovery<<epoch<<','<<n<<','<<restored<<','<<Simulator::Now().GetNanoSeconds()<<','
                            <<m_aps[n]->GetPhy()->GetState()->GetState()<<','<<m_stas[n]->GetPhy()->GetState()->GetState()<<','
                            <<DynamicCast<StaWifiMac>(m_stas[n]->GetMac())->IsAssociated()<<','<<unsigned(m_stas[n]->GetPhy()->GetChannelNumber())<<'\n';
                    }
                    output<<epoch<<','<<started<<','<<Simulator::Now().GetNanoSeconds()<<",1,"<<nonempty<<",0,0,0\n";output.flush();
                    m_scanRecovery.flush();m_measuring=false;ReleaseScan(epoch);
                    if(m_fresh)m_fresh->Publish();
                });
            });return;
        }
        output<<epoch<<','<<started<<','<<now<<",0,"<<nonempty<<','<<busy<<','<<response<<','<<guard<<'\n';output.flush();
        ReleaseScan(epoch);
        if(m_fresh)m_fresh->Unavailable(epoch);
        CloseTracking(epoch);
    }
    void ScanRetune(uint64_t epoch,const std::vector<uint8_t>& channels) {
        for(unsigned n=0;n<m_sources.size();++n) {
            NS_ABORT_MSG_IF(!ScanRetunable(m_aps[n]->GetPhy()->GetState()->GetState())||!ScanRetunable(m_stas[n]->GetPhy()->GetState()->GetState())||
                !ResponseTimerIdle(m_aps[n])||!ResponseTimerIdle(m_stas[n]),"scan retune radio unsafe");
            QueueCheckpoint(n,epoch,"before_switch");
            auto settings=StringValue("{"+std::to_string(36+4*channels[n])+",20,BAND_5GHZ,0}");
            m_aps[n]->GetPhy()->SetAttribute("ChannelSettings",settings);m_stas[n]->GetPhy()->SetAttribute("ChannelSettings",settings);
            QueueCheckpoint(n,epoch,"after_switch");
        }
    }
    void ReleaseScan(uint64_t epoch) {
        for(unsigned n=0;n<m_sources.size();++n) {
            QueueCheckpoint(n,epoch,"before_resume");
            NS_ABORT_MSG_IF(!m_sources[n]->ResumeAfterChannelSwitch(),"scan source refused resume");
            m_demandLedger.EndChannelPause(n);QueueCheckpoint(n,epoch,"after_resume");
            m_aps[n]->GetMac()->GetChannelAccessManager()->SetMeasurementHold(false);
            m_stas[n]->GetMac()->GetChannelAccessManager()->SetMeasurementHold(false);
            Simulator::Schedule(MilliSeconds(n),[ap=m_aps[n]] {ap->GetMac()->SetAttribute("BeaconGeneration",BooleanValue(true));});
        }
        m_scanInProgress=false;
    }
    void Tick()
    {
        ValidateRecoveryDomain(m_allWifi,m_waveform,m_delay);
        const bool actuate=(m_schedule=="two_window" && (m_tick==2 || m_tick==3)) ||
                           (m_schedule=="isolated" && m_tick==2);
        for(uint32_t node=0;node<m_stas.size();++node)
        {
            m_telemetry.ControlOpportunity(node);
            const auto oldChannel=m_stas[node]->GetPhy()->GetChannelNumber();
            if(actuate && !m_pending[node].active)
            {
                Begin(node,m_tick,oldChannel,36+4*(((oldChannel-36)/4+1)%4));
            }
            else
            {
                Trace(node,m_tick,oldChannel,oldChannel,false,
                      m_pending[node].active?"pending":"opportunity",0,false,false,false);
            }
        }
        m_output.flush();
        ValidateRecoveryDomain(m_allWifi,m_waveform,m_delay);
        ++m_tick;
        if(Simulator::Now().GetSeconds()+m_interval<m_stop-1e-9)
            Simulator::Schedule(Seconds(m_interval),&ChannelController::Tick,this);
    }
    struct Pending
    {
        bool active=false;
        uint64_t opportunity=0;
        uint16_t oldChannel=0;
        uint16_t targetChannel=0;
        int64_t pauseAt=0,quietAt=-1;
        int64_t quiescenceDeadline=0;
    };

    static bool QueueEmpty(Ptr<WifiNetDevice> device)
    {
        auto mac=device->GetMac();
        if(auto txop=mac->GetTxop();txop && txop->GetWifiMacQueue()->GetNPackets())return false;
        for(auto ac:{AC_BE,AC_BK,AC_VI,AC_VO})
        {
            if(auto queue=mac->GetTxopQueue(ac);queue && queue->GetNPackets())return false;
        }
        return true;
    }

    static bool ResponseTimerIdle(Ptr<WifiNetDevice> device)
    {
        return !device->GetMac()->GetFrameExchangeManager()->GetWifiTxTimer().IsRunning();
    }

    Time ResponseGuard(uint32_t node) const
    {
        auto phy=m_stas[node]->GetPhy();
        auto manager=m_stas[node]->GetRemoteStationManager();
        auto control=manager->GetRtsTxVector(Mac48Address::GetBroadcast(),phy->GetChannelWidth());
        const Time controlDuration=WifiPhy::CalculateTxDuration(GetAckSize(),control,phy->GetPhyBand());
        // In fixed 802.11a, deferred immediate ACK/CTS/next-fragment callbacks are
        // scheduled at SIFS. Include a base-rate control frame, DIFS, and certified
        // propagation coverage; any later PHY activity restarts this continuous guard.
        return 2*phy->GetSifs()+controlDuration+2*phy->GetSlot()+RecoveryPropagationBound();
    }

    void Begin(uint32_t node,uint64_t opportunity,uint16_t oldChannel,uint16_t targetChannel)
    {
        auto& pending=m_pending[node];
        pending.active=true;pending.opportunity=opportunity;pending.oldChannel=oldChannel;
        pending.targetChannel=targetChannel;
        pending.pauseAt=Simulator::Now().GetNanoSeconds();pending.quietAt=-1;
        pending.quiescenceDeadline=pending.pauseAt+MilliSeconds(200).GetNanoSeconds();
        QueueCheckpoint(node,opportunity,"before_pause");
        m_demandLedger.BeginChannelPause(node);
        NS_ABORT_MSG_IF(!m_sources[node]->PauseForChannelSwitch(),"source refused explicit channel pause");
        QueueCheckpoint(node,opportunity,"after_pause");
        Trace(node,opportunity,oldChannel,oldChannel,false,"pause_requested",0,false,false,false);
        Simulator::Schedule(MicroSeconds(50),&ChannelController::Poll,this,node);
    }

    void Poll(uint32_t node)
    {
        auto& pending=m_pending[node];
        if(!pending.active)return;
        const int64_t now=Simulator::Now().GetNanoSeconds();
        const bool queuesEmpty=QueueEmpty(m_aps[node])&&QueueEmpty(m_stas[node]);
        const bool physIdle=m_aps[node]->GetPhy()->IsStateIdle()&&m_stas[node]->GetPhy()->IsStateIdle();
        const bool timersIdle=ResponseTimerIdle(m_aps[node])&&ResponseTimerIdle(m_stas[node]);
        const int64_t latestActivity=std::max(m_apActivity[node]->lastActivityEnd,
                                              m_staActivity[node]->lastActivityEnd);
        if(queuesEmpty&&physIdle&&timersIdle&&now-latestActivity>=ResponseGuard(node).GetNanoSeconds())
        {
            const auto newChannel=pending.targetChannel;
            const auto settings="{"+std::to_string(newChannel)+", 20, BAND_5GHZ, 0}";
            QueueCheckpoint(node,pending.opportunity,"before_switch");
            m_aps[node]->GetPhy()->SetAttribute("ChannelSettings",StringValue(settings));
            m_stas[node]->GetPhy()->SetAttribute("ChannelSettings",StringValue(settings));
            QueueCheckpoint(node,pending.opportunity,"after_switch");
            Trace(node,pending.opportunity,pending.oldChannel,newChannel,false,"switching_started",
                  now-pending.pauseAt,queuesEmpty,physIdle,timersIdle);
            Simulator::Schedule(std::max(m_aps[node]->GetPhy()->GetDelayUntilIdle(),
                                         m_stas[node]->GetPhy()->GetDelayUntilIdle()),
                                &ChannelController::FinishSwitch,this,node,newChannel);
            return;
        }
        if(now>=pending.quiescenceDeadline)
        {
            const auto pauseDuration=now-pending.pauseAt;
            QueueCheckpoint(node,pending.opportunity,"before_resume");
            NS_ABORT_MSG_IF(!m_sources[node]->ResumeAfterChannelSwitch(),"source refused rejected resume");
            m_demandLedger.EndChannelPause(node);
            QueueCheckpoint(node,pending.opportunity,"after_resume");
            Trace(node,pending.opportunity,pending.oldChannel,pending.oldChannel,false,"rejected",
                  pauseDuration,queuesEmpty,physIdle,timersIdle);
            pending.active=false;
            Complete(node,threeway::ManagementHeader::REJECTED);
            return;
        }
        Simulator::Schedule(MicroSeconds(50),&ChannelController::Poll,this,node);
    }

    void FinishSwitch(uint32_t node,uint16_t newChannel)
    {
        auto& pending=m_pending[node];
        NS_ABORT_MSG_IF(!pending.active,"finish without pending channel switch");
        if(!m_aps[node]->GetPhy()->IsStateIdle()||!m_stas[node]->GetPhy()->IsStateIdle())
        {
            Simulator::Schedule(std::max(m_aps[node]->GetPhy()->GetDelayUntilIdle(),
                                         m_stas[node]->GetPhy()->GetDelayUntilIdle()),
                                &ChannelController::FinishSwitch,this,node,newChannel);
            return;
        }
        NS_ABORT_MSG_IF(m_aps[node]->GetPhy()->GetChannelNumber()!=newChannel||
                        m_stas[node]->GetPhy()->GetChannelNumber()!=newChannel,
                        "channel switch did not reach requested channel");
        const auto pauseDuration=Simulator::Now().GetNanoSeconds()-pending.pauseAt;
        QueueCheckpoint(node,pending.opportunity,"before_resume");
        NS_ABORT_MSG_IF(!m_sources[node]->ResumeAfterChannelSwitch(),"source refused channel resume");
        m_demandLedger.EndChannelPause(node);
        QueueCheckpoint(node,pending.opportunity,"after_resume");
        Trace(node,pending.opportunity,pending.oldChannel,newChannel,true,"changed",
              pauseDuration,true,true,true);
        pending.active=false;
        ValidateRecoveryDomain(m_allWifi,m_waveform,m_delay);
        Complete(node,threeway::ManagementHeader::OK);
    }

    void QueueCheckpoint(uint32_t node,uint64_t epoch,const char* phase) {
        if(m_scanInProgress&&m_schedule=="external")epoch+=1000000;
        if(m_queueContinuity)m_queueContinuity->Checkpoint(node,epoch,phase);
    }
    void Complete(uint32_t node,uint8_t status) {
        if(m_done[node]){auto done=std::move(m_done[node]);m_done[node]={};done(status);}
    }

    void Trace(uint32_t node,uint64_t opportunity,uint16_t oldChannel,uint16_t newChannel,
               bool changed,const char* status,int64_t pauseDuration,bool queuesEmpty,
               bool physIdle,bool timersIdle)
    {
        m_output<<Simulator::Now().GetSeconds()<<','<<node<<','<<opportunity<<','
                <<static_cast<unsigned>(oldChannel)<<','<<static_cast<unsigned>(newChannel)<<','
                <<changed<<','<<status<<','<<pauseDuration<<','<<queuesEmpty<<','<<physIdle<<','
                <<timersIdle<<','<<ResponseGuard(node).GetNanoSeconds()<<','<<m_schedule<<'\n';
        m_output.flush();
    }
    std::vector<Ptr<WifiNetDevice>> m_aps,m_stas;
    NetDeviceContainer m_allWifi;
    Ptr<WaveformGenerator> m_waveform;
    Ptr<ConstantSpeedPropagationDelayModel> m_delay;
    CorrectedTelemetry& m_telemetry;
    TelemetryRiskController& m_demandLedger;
    std::vector<Ptr<OnOffApplication>> m_sources;
    double m_interval,m_stop;
    std::string m_schedule;
    std::ofstream m_output,m_scanDiagnostic,m_scanRecovery;
    uint64_t m_tick=0;
    std::vector<Pending> m_pending;
    bool m_measuring=false,m_scanInProgress=false;
    FreshSounding* m_fresh=nullptr;
    bool m_windowTracking=false;uint64_t m_trackingEpoch=0;std::ofstream m_tracking;
    std::vector<std::shared_ptr<ChannelQuiescenceListener>> m_apActivity,m_staActivity;
    std::vector<threeway::ManagementEpochs::Done> m_done;
    QueueContinuity* m_queueContinuity;
};

static bool
HasScenario(const std::string& scenario, const std::string& name)
{
    return scenario == name || scenario == "all";
}

static std::vector<Vector>
ReadPositions(const std::string& path, uint32_t numNodes)
{
    std::vector<Vector> positions;
    if (path.empty())
    {
        return positions;
    }

    std::ifstream input(path);
    if (!input)
    {
        NS_FATAL_ERROR("cannot open positions file: " << path);
    }
    std::string line;
    while (std::getline(input, line))
    {
        if (line.empty())
        {
            continue;
        }
        std::stringstream row(line);
        std::string token;
        std::vector<double> values;
        while (std::getline(row, token, ','))
        {
            values.push_back(std::stod(token));
        }
        if (values.size() == 2)
        {
            values.push_back(0.0);
        }
        if (values.size() != 3)
        {
            NS_FATAL_ERROR("each position row must contain x,y[,z]");
        }
        positions.emplace_back(values[0], values[1], values[2]);
    }
    if (positions.size() != numNodes)
    {
        NS_FATAL_ERROR("positions file must contain exactly " << numNodes << " rows");
    }
    return positions;
}

static std::vector<double>
ReadLoads(const std::string& path, uint32_t numNodes)
{
    std::vector<double> loads;
    if (path.empty())
    {
        return loads;
    }

    std::ifstream input(path);
    if (!input)
    {
        NS_FATAL_ERROR("cannot open loads file: " << path);
    }
    std::string line;
    while (std::getline(input, line))
    {
        if (!line.empty())
        {
            loads.push_back(std::stod(line));
        }
    }
    if (loads.size() != numNodes)
    {
        NS_FATAL_ERROR("loads file must contain exactly " << numNodes << " rows");
    }
    for (double load : loads)
    {
        if (!std::isfinite(load) || load < 0.0)
        {
            NS_FATAL_ERROR("load values must be finite and non-negative");
        }
    }
    return loads;
}

static std::vector<std::vector<double>>
ReadQValues(const std::string& path, uint32_t numNodes, uint32_t numChannels)
{
    std::vector<std::vector<double>> values;
    if (path.empty())
    {
        return values;
    }
    std::ifstream input(path);
    if (!input)
    {
        NS_FATAL_ERROR("cannot open q-values file: " << path);
    }
    std::string line;
    while (std::getline(input, line))
    {
        if (line.empty())
        {
            continue;
        }
        std::stringstream row(line);
        std::string token;
        std::vector<double> current;
        while (std::getline(row, token, ','))
        {
            current.push_back(std::stod(token));
        }
        if (current.size() != numChannels)
        {
            NS_FATAL_ERROR("each q-values row must contain exactly " << numChannels << " values");
        }
        values.push_back(current);
    }
    if (values.size() != numNodes)
    {
        NS_FATAL_ERROR("q-values file must contain exactly " << numNodes << " rows");
    }
    return values;
}

static double
PairwiseRssi(const Vector& a, const Vector& b, double interferenceRange)
{
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double distance = std::sqrt(dx * dx + dy * dy);
    return distance < interferenceRange ? 1.0 - distance / interferenceRange : 0.0;
}

static double
PressureFor(const std::vector<uint32_t>& assignment,
            const std::vector<Vector>& positions,
            const std::vector<double>& loads,
            uint32_t node,
            uint32_t channel,
            double interferenceRange,
            uint32_t unassigned)
{
    double pressure = 0.0;
    for (uint32_t j = 0; j < assignment.size(); ++j)
    {
        if (j == node || assignment[j] == unassigned || assignment[j] != channel)
        {
            continue;
        }
        pressure += PairwiseRssi(positions[node], positions[j], interferenceRange) * loads[j];
    }
    return pressure;
}

static void
ApplyExecutionPressureMode(std::vector<uint32_t>& assignment,
                            const std::vector<std::vector<double>>& qValues,
                            const std::vector<Vector>& positions,
                            const std::vector<double>& loads,
                            uint32_t numChannels,
                            const std::string& mode,
                            double noiseStd,
                            double emaAlpha,
                            double reservationWeight,
                            double interferenceRange,
                            uint32_t runNumber)
{
    if (qValues.empty() || positions.size() != assignment.size() || loads.size() != assignment.size() ||
        mode == "input")
    {
        return;
    }
    const uint32_t n = assignment.size();
    const uint32_t unassigned = numChannels;
    std::vector<uint32_t> qGreedy(n, 0);
    std::vector<double> spread(n, 1.0);
    for (uint32_t i = 0; i < n; ++i)
    {
        qGreedy[i] = static_cast<uint32_t>(std::distance(
            qValues[i].begin(), std::max_element(qValues[i].begin(), qValues[i].end())));
        const auto range = std::minmax_element(qValues[i].begin(), qValues[i].end());
        spread[i] = std::max(*range.second - *range.first, 1e-6);
    }
    if (mode == "argmax_q")
    {
        assignment = qGreedy;
        return;
    }

    std::mt19937 generator(1000u + runNumber);
    std::normal_distribution<double> noise(0.0, std::max(0.0, noiseStd));
    std::vector<uint32_t> tentative = mode == "reservation"
                                          ? std::vector<uint32_t>(n, unassigned)
                                          : qGreedy;
    const std::vector<uint32_t> orderState = qGreedy;
    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return PressureFor(orderState, positions, loads, a, orderState[a], interferenceRange, unassigned) >
               PressureFor(orderState, positions, loads, b, orderState[b], interferenceRange, unassigned);
    });
    const double eta = 0.05;
    for (uint32_t node : order)
    {
        std::vector<double> estimate(numChannels, 0.0);
        for (uint32_t channel = 0; channel < numChannels; ++channel)
        {
            const double raw = PressureFor(tentative, positions, loads, node, channel,
                                            interferenceRange, unassigned);
            if (mode == "ema")
            {
                double smoothed = raw;
                for (int sample = 0; sample < 5; ++sample)
                {
                    smoothed = (1.0 - emaAlpha) * smoothed + emaAlpha * (raw + noise(generator));
                }
                estimate[channel] = std::max(0.0, smoothed);
            }
            else
            {
                estimate[channel] = std::max(0.0, raw + (mode == "noisy" ? noise(generator) : 0.0));
            }
        }
        double bestScore = -std::numeric_limits<double>::infinity();
        uint32_t bestChannel = tentative[node];
        for (uint32_t channel = 0; channel < numChannels; ++channel)
        {
            double score = qValues[node][channel];
            score -= 8.0 * spread[node] * std::max(0.0, estimate[channel] - eta);
            if (channel != qGreedy[node])
            {
                score -= 0.1 * spread[node];
            }
            if (mode == "reservation")
            {
                score -= reservationWeight * estimate[channel];
            }
            if (score > bestScore)
            {
                bestScore = score;
                bestChannel = channel;
            }
        }
        tentative[node] = bestChannel;
    }
    assignment = tentative;
}

int
main(int argc, char* argv[])
{
    uint32_t numNodes = 15;
    uint32_t numChannels = 4;
    double simTime = 10.0;
    uint32_t payloadSize = 1200;
    std::string dataRate = "2Mbps";
    std::string assignmentText = "0,1,2,3,0,1,2,3,0,1,2,3,0,1,2";
    std::string outputPath = "results/wifi_mesh_packet_validation.csv";
    std::string flowMonitorOutput = "results/wifi_mesh_packet_validation.flowmon.xml";
    std::string positionsFile;
    std::string loadsFile;
    std::string qValuesFile;
    std::string scenario = "baseline";
    std::string executionMode = "input";
    std::string channelControlSchedule = "quiet";
    double pressureNoiseStd = 0.05;
    double emaAlpha = 0.3;
    double reservationWeight = 1.0;
    double telemetryInterval = 0.7;
    double retryRiskThreshold = 0.05;
    double phyErrorRiskThreshold = 0.10;
    double busyRiskThreshold = 0.75;
    double backlogRiskThreshold = 0.05;
    double globalBusyRiskThreshold = 0.46;
    double globalBacklogRiskThreshold = 0.10;
    double driftRiskThreshold = 0.25;
    double collisionReleaseRatio = 0.50;
    double pacingFloor = 0.70;
    double capacityReleaseRatio = 0.50;
    uint32_t minGuardDwellWindows = 2;
    uint32_t guardActivationWindows = 2;
    bool scaleRateByLoad = false;
    bool forceHiddenCollision = false;
    bool enableRtsCts = false;
    bool enableTelemetryGuard = false;
    bool enableCapacityPacing = false;
    bool enableAdjacentExcitation = true;
    bool enableChannelController = false;
    bool useV31TrafficProfile = false;
    bool enableSounding = false;
    bool enablePolicyBridge = false;
    bool enablePacketCohort = false;
    bool enableQueueContinuity = false;
    bool probeScanQuiescence = false;
    bool probeFreshSounding = false;
    bool compactSounding = false;std::string trackingMode="continuous";int64_t developmentStopAtNs=0;
    std::string demandScheduleFile;
    uint64_t managementDropAckEpoch = 0;
    int managementDropAckNode = -1;
    uint32_t runNumber = 1;
    int requiredCpu = -1;

    CommandLine cmd(__FILE__);
    cmd.AddValue("probeScanQuiescence","Static mobility scan prerequisite only",probeScanQuiescence);
    cmd.AddValue("probeFreshSounding","Repeated ReceivedPsd acquisition diagnostic; no policy release",probeFreshSounding);
    cmd.AddValue("trackingMode","continuous/window receive-history lifecycle",trackingMode);
    cmd.AddValue("developmentStopAtNs","Diagnostic early stop; no service score",developmentStopAtNs);
    cmd.AddValue("compactSounding","Fixed250us pilot slots/22ms acquisition; diagnostic only",compactSounding);
    cmd.AddValue("numNodes", "Number of AP/STA pairs", numNodes);
    cmd.AddValue("numChannels", "Number of available channels", numChannels);
    cmd.AddValue("simTime", "Simulation time in seconds", simTime);
    cmd.AddValue("payloadSize", "UDP payload size in bytes", payloadSize);
    cmd.AddValue("dataRate", "Offered rate per STA", dataRate);
    cmd.AddValue("assignment", "Comma-separated channel assignment", assignmentText);
    cmd.AddValue("output", "CSV output path", outputPath);
    cmd.AddValue("flowMonitorOutput", "FlowMonitor XML output path", flowMonitorOutput);
    cmd.AddValue("positionsFile", "CSV file with one x,y[,z] row per AP", positionsFile);
    cmd.AddValue("loadsFile", "File with one normalized load value per STA", loadsFile);
    cmd.AddValue("qValuesFile", "CSV file with per-node Q-values", qValuesFile);
    cmd.AddValue("demandScheduleFile", "Prospective complete continuous-positive rate snapshots", demandScheduleFile);
    cmd.AddValue("scenario",
                 "baseline|mobility|hidden_node|burst|adjacent_channel|all",
                 scenario);
    cmd.AddValue("scaleRateByLoad", "Scale each STA offered rate by its load", scaleRateByLoad);
    cmd.AddValue("forceHiddenCollision",
                 "Diagnostic mode: override policy and force hidden pair onto one channel",
                 forceHiddenCollision);
    cmd.AddValue("enableRtsCts",
                 "Force RTS/CTS for data frames by setting the RTS threshold to zero",
                 enableRtsCts);
    cmd.AddValue("enableTelemetryGuard",
                 "Use telemetry risk probes to control RTS/CTS without scenario labels",
                 enableTelemetryGuard);
    cmd.AddValue("enableCapacityPacing",
                 "Use telemetry capacity risk to pace application demand",
                 enableCapacityPacing);
    cmd.AddValue("enableAdjacentExcitation",
                 "Enable the external adjacent-channel waveform environment stimulus",
                 enableAdjacentExcitation);
    cmd.AddValue("enableChannelController",
                 "Execute the eligible channel controller callback at each telemetry boundary",
                 enableChannelController);
    cmd.AddValue("channelControlSchedule",
                 "quiet|two_window|isolated; independent diagnostic actuation schedule",
                 channelControlSchedule);
    cmd.AddValue("pacingFloor", "Minimum pacing factor in [0.70, 1.00]", pacingFloor);
    cmd.AddValue("capacityReleaseRatio",
                 "Release pacing when capacity risk falls below this normalized ratio",
                 capacityReleaseRatio);
    cmd.AddValue("useV31TrafficProfile",
                 "Use continuous non-burst traffic and equal-average-load burst traffic",
                 useV31TrafficProfile);
    cmd.AddValue("telemetryInterval",
                 "Controller telemetry sampling interval in seconds",
                 telemetryInterval);
    cmd.AddValue("retryRiskThreshold", "Retry-rate activation threshold", retryRiskThreshold);
    cmd.AddValue("phyErrorRiskThreshold",
                 "PHY receive-error activation threshold",
                 phyErrorRiskThreshold);
    cmd.AddValue("busyRiskThreshold", "Channel-busy activation threshold", busyRiskThreshold);
    cmd.AddValue("backlogRiskThreshold",
                 "Application backlog activation threshold",
                 backlogRiskThreshold);
    cmd.AddValue("globalBusyRiskThreshold",
                 "Controller-wide channel-busy activation threshold",
                 globalBusyRiskThreshold);
    cmd.AddValue("globalBacklogRiskThreshold",
                 "Controller-wide backlog activation threshold",
                 globalBacklogRiskThreshold);
    cmd.AddValue("driftRiskThreshold",
                 "Relative telemetry-change activation threshold",
                 driftRiskThreshold);
    cmd.AddValue("minGuardDwellWindows",
                 "Minimum telemetry windows between RTS/CTS state changes",
                 minGuardDwellWindows);
    cmd.AddValue("guardActivationWindows",
                 "Consecutive collision-risk windows required before enabling RTS/CTS",
                 guardActivationWindows);
    cmd.AddValue("collisionReleaseRatio",
                 "Release RTS/CTS when collision risk falls below this normalized ratio",
                 collisionReleaseRatio);
    cmd.AddValue("executionMode", "input|argmax_q|noisy|ema|reservation", executionMode);
    cmd.AddValue("pressureNoiseStd", "Standard deviation of execution pressure noise", pressureNoiseStd);
    cmd.AddValue("emaAlpha", "EMA update coefficient for pressure estimates", emaAlpha);
    cmd.AddValue("reservationWeight", "Reservation pressure penalty weight", reservationWeight);
    cmd.AddValue("runNumber", "ns-3 random-stream run number", runNumber);
    cmd.AddValue("enableSounding", "Common pre-traffic receiver sounding", enableSounding);
    cmd.AddValue("enablePolicyBridge", "Read policy vectors from stdin, use actual management packets", enablePolicyBridge);
    cmd.AddValue("enablePacketCohort", "Evaluator-only application UID delay/loss/censoring ledger", enablePacketCohort);
    cmd.AddValue("enableQueueContinuity", "Evaluator-only queue/inflight UID continuity trace", enableQueueContinuity);
    cmd.AddValue("managementDropAckEpoch", "Development fault injection: lose one epoch ACK", managementDropAckEpoch);
    cmd.AddValue("managementDropAckNode", "Development fault injection: blocked AP", managementDropAckNode);
    cmd.AddValue("requiredCpu", "Optional qualified per-process CPU affinity requirement", requiredCpu);
    cmd.Parse(argc, argv);

    if (requiredCpu != -1)
    {
        cpu_set_t cpus;
        CPU_ZERO(&cpus);
        NS_ABORT_MSG_IF(requiredCpu != 2 || sched_getaffinity(0, sizeof(cpus), &cpus) != 0 ||
                            CPU_COUNT(&cpus) != 1 || !CPU_ISSET(2, &cpus),
                        "actual native CPU affinity differs from campaign requirement");
        std::cout << "THREEWAY_RESOURCE {\"pid\":" << getpid() << ",\"cpus\":[2]}" << std::endl;
    }

    const std::vector<std::string> validScenarios = {
        "baseline", "mobility", "hidden_node", "burst", "adjacent_channel", "all"};
    if (std::find(validScenarios.begin(), validScenarios.end(), scenario) == validScenarios.end())
    {
        NS_FATAL_ERROR("unknown scenario: " << scenario);
    }
    if (telemetryInterval <= 0.0 || telemetryInterval >= simTime - 1.0)
    {
        NS_FATAL_ERROR("telemetryInterval must be positive and leave at least one traffic window");
    }
    if (!std::isfinite(pacingFloor) || pacingFloor < 0.70 || pacingFloor > 1.00)
    {
        NS_FATAL_ERROR("pacingFloor must be in [0.70, 1.00]");
    }
    if (numNodes!=15 || numChannels!=4 || simTime!=10.0 || payloadSize!=1200 ||
        telemetryInterval!=0.7 || !scaleRateByLoad || !useV31TrafficProfile ||
        positionsFile.empty() || loadsFile.empty() || qValuesFile.empty())
    {
        NS_FATAL_ERROR("production recovery certificate configuration domain");
    }
    NS_ABORT_MSG_IF((trackingMode!="continuous"&&trackingMode!="window")||(developmentStopAtNs&& (developmentStopAtNs<1700000000||developmentStopAtNs>=10000000000LL)),"invalid integration mode/early stop");
    NS_ABORT_MSG_IF(trackingMode=="window"&&(!compactSounding||!probeFreshSounding||!enablePolicyBridge),"window lifecycle requires compact live bridge");
    RngSeedManager::SetRun(runNumber);
    auto assignment = ParseAssignment(assignmentText, numNodes, numChannels);
    const bool mobilityScenario = HasScenario(scenario, "mobility");
    const bool hiddenNodeScenario = HasScenario(scenario, "hidden_node");
    const bool burstScenario = HasScenario(scenario, "burst");
    NS_ABORT_MSG_IF(probeFreshSounding&&(!probeScanQuiescence||!enableSounding),"fresh probe requires scan and initial setup");
    NS_ABORT_MSG_IF(compactSounding&&!probeFreshSounding,"compact schedule requires fresh sounding");
    NS_ABORT_MSG_IF(probeScanQuiescence&&((!probeFreshSounding&&(!mobilityScenario||burstScenario))||!enableChannelController||
                   (channelControlSchedule!="quiet"&&!(probeFreshSounding&&enablePolicyBridge&&channelControlSchedule=="external"))),
                   "scan probe requires static continuous mobility path");
    NS_ABORT_MSG_IF(enablePolicyBridge&&mobilityScenario&&!probeFreshSounding,
                   "mobility policy requires qualified fresh receiver sounding and bounded acquisition age");
    NS_ABORT_MSG_IF(!demandScheduleFile.empty()&&(burstScenario||enableTelemetryGuard||enableCapacityPacing||executionMode!="input"),
                    "piecewise demand requires unpaced continuous-On input execution");
    NS_ABORT_MSG_IF(enablePolicyBridge&&(!enableSounding||!enableChannelController||channelControlSchedule!="external"||
                   (burstScenario&&!probeFreshSounding)||enableTelemetryGuard||enableCapacityPacing||executionMode!="input"),
                   "policy bridge requires qualified common continuous-demand path");
    NS_ABORT_MSG_IF(channelControlSchedule=="external"&&!enablePolicyBridge,"external schedule requires policy bridge");
    const bool adjacentExcitationScenario = HasScenario(scenario, "adjacent_channel");
    if(enableChannelController && channelControlSchedule!="quiet" && burstScenario && !probeFreshSounding)
    {
        NS_FATAL_ERROR("quiescent channel actuation supports continuous-On traffic only");
    }
    if (hiddenNodeScenario && forceHiddenCollision && numNodes >= 2)
    {
        // Optional diagnostic stress only. Comparative policy evaluation keeps
        // this false so a DCA policy can mitigate the hidden pair by separating it.
        assignment[1] = assignment[0];
    }
    const auto importedPositions = ReadPositions(positionsFile, numNodes);
    const auto importedLoads = ReadLoads(loadsFile, numNodes);
    const auto qValues = ReadQValues(qValuesFile, numNodes, numChannels);
    ApplyExecutionPressureMode(assignment,
                               qValues,
                               importedPositions,
                               importedLoads,
                               numChannels,
                               executionMode,
                               pressureNoiseStd,
                               emaAlpha,
                               reservationWeight,
                               60.0,
                               runNumber);
    {
        std::ofstream assignmentOutput(outputPath + ".assignment.csv");
        assignmentOutput << "node,channel,executionMode\n";
        for (uint32_t i = 0; i < assignment.size(); ++i)
        {
            assignmentOutput << i << ',' << assignment[i] << ',' << executionMode << '\n';
        }
    }

    std::ofstream geometryAudit(outputPath + ".geometry.csv");
    geometryAudit.precision(17);
    geometryAudit << "pair,role,x,y,z,channel\n";
    NS_ABORT_MSG_IF(!geometryAudit, "geometry audit open failed");
    NodeContainer aps;
    NodeContainer stas;
    aps.Create(numNodes);
    stas.Create(numNodes);

    MobilityHelper mobility;
    Ptr<ListPositionAllocator> apPositions = CreateObject<ListPositionAllocator>();
    Ptr<ListPositionAllocator> staPositions = CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < numNodes; ++i)
    {
        Vector base = importedPositions.empty()
                          ? Vector(15.0 + 20.0 * (i % 5),
                                   15.0 + 20.0 * (i / 5),
                                   1.5)
                          : importedPositions[i];
        if (hiddenNodeScenario && i == 0)
        {
            base = Vector(50.0, 50.0, 1.5);
        }
        else if (hiddenNodeScenario && i == 1)
        {
            base = Vector(50.0, 54.0, 1.5);
        }
        apPositions->Add(base);
        if (hiddenNodeScenario && i == 0)
        {
            staPositions->Add(Vector(20.0, 50.0, 1.0));
        }
        else if (hiddenNodeScenario && i == 1)
        {
            staPositions->Add(Vector(80.0, 54.0, 1.0));
        }
        else
        {
            staPositions->Add(Vector(base.x + 2.0, base.y + 1.0, 1.0));
        }
    }
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.SetPositionAllocator(apPositions);
    mobility.Install(aps);
    if (mobilityScenario)
    {
        mobility.SetMobilityModel("ns3::WaypointMobilityModel");
    }
    mobility.SetPositionAllocator(staPositions);
    mobility.Install(stas);
    if(mobilityScenario)for(unsigned n=0;n<stas.GetN();++n) {
        auto model=stas.Get(n)->GetObject<WaypointMobilityModel>();auto p=model->GetPosition();
        model->AddWaypoint(Waypoint(Seconds(0),p));
        model->AddWaypoint(Waypoint(Seconds(1),p));
        model->AddWaypoint(Waypoint(Seconds(5),Vector(p.x+8,p.y,p.z)));
        model->AddWaypoint(Waypoint(Seconds(9),p));
        model->AddWaypoint(Waypoint(Seconds(10),p));
    }

    Ptr<MultiModelSpectrumChannel> sharedSpectrumChannel =
        CreateObject<AcquisitionSpectrumChannel>();
    Ptr<FriisPropagationLossModel> propagation = CreateObject<FriisPropagationLossModel>();
    propagation->SetAttribute("Frequency", DoubleValue(5.150e9));
    propagation->SetAttribute("SystemLoss", DoubleValue(1.0));
    propagation->SetAttribute("MinLoss", DoubleValue(0.0));
    sharedSpectrumChannel->AddPropagationLossModel(propagation);
    if(hiddenNodeScenario) {
        // Fixed peer-only obstruction; desired and interfering STA-to-AP paths stay physical.
        auto peerObstruction=CreateObject<MatrixPropagationLossModel>();
        peerObstruction->SetDefaultLoss(0);
        peerObstruction->SetLoss(stas.Get(0)->GetObject<MobilityModel>(),
                                 stas.Get(1)->GetObject<MobilityModel>(),40,true);
        sharedSpectrumChannel->AddPropagationLossModel(peerObstruction);
    }
    Ptr<ConstantSpeedPropagationDelayModel> propagationDelay =
        CreateObject<ConstantSpeedPropagationDelayModel>();
    sharedSpectrumChannel->SetPropagationDelayModel(propagationDelay);

    NodeContainer adjacentEmitter;
    adjacentEmitter.Create(1);
    // Avoid MobilityHelper's unused default random variables shifting Wi-Fi streams.
    Ptr<ConstantPositionMobilityModel> emitterMobility = CreateObject<ConstantPositionMobilityModel>();
    emitterMobility->SetPosition(Vector(75.0, 75.0, 1.5));
    adjacentEmitter.Get(0)->AggregateObject(emitterMobility);
    WaveformGeneratorHelper waveformGenerator;
    waveformGenerator.SetChannel(sharedSpectrumChannel);
    waveformGenerator.SetTxPowerSpectralDensity(
        WifiSpectrumValueHelper::CreateOfdmTxPowerSpectralDensity(MHz_u{5260},
                                                                   MHz_u{20},
                                                                   DbmToW(dBm_u{16.0206}),
                                                                   MHz_u{20}));
    waveformGenerator.SetPhyAttribute("Period", TimeValue(MilliSeconds(100)));
    waveformGenerator.SetPhyAttribute("DutyCycle", DoubleValue(1.0));
    NetDeviceContainer adjacentEmitterDevice = waveformGenerator.Install(adjacentEmitter);
    Ptr<WaveformGenerator> adjacentWaveform =
        adjacentEmitterDevice.Get(0)->GetObject<NonCommunicatingNetDevice>()->GetPhy()->GetObject<
            WaveformGenerator>();
    NS_ABORT_MSG_IF(!adjacentWaveform, "expected adjacent-channel WaveformGenerator");

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211a);
    wifi.SetRemoteStationManager("ns3::IdealWifiManager",
                                 "RtsCtsThreshold",
                                 UintegerValue(enableRtsCts ? 0 : 65535));

    TelemetryRiskController telemetryController(numNodes,
                                                telemetryInterval,
                                                simTime - 0.1,
                                                outputPath + ".telemetry.csv",
                                                enableTelemetryGuard,
                                                enableCapacityPacing,
                                                enableRtsCts,
                                                retryRiskThreshold,
                                                phyErrorRiskThreshold,
                                                busyRiskThreshold,
                                                backlogRiskThreshold,
                                                globalBusyRiskThreshold,
                                                globalBacklogRiskThreshold,
                                                driftRiskThreshold,
                                                minGuardDwellWindows,
                                                guardActivationWindows,
                                                collisionReleaseRatio,
                                                capacityReleaseRatio,
                                                pacingFloor);
    g_telemetryController = &telemetryController;
    NS_ABORT_MSG_IF(enableTelemetryGuard || enableCapacityPacing, "corrected candidate is observation-only");
    CorrectedTelemetry corrected(numNodes, telemetryInterval, simTime, outputPath + ".corrected.csv");
    g_corrected = &corrected;
    std::unique_ptr<PacketCohort> packetCohort;
    if(enablePacketCohort) {
        packetCohort=std::make_unique<PacketCohort>(numNodes,simTime,outputPath+".packet-cohort.csv");
        g_packetCohort=packetCohort.get();
    }
    OfflineEnergyRecorder offlineEnergyRecorder(numNodes,
                                                adjacentEmitter.Get(0)->GetId(),
                                                outputPath + ".offline-energy.csv");
    g_offlineEnergyRecorder = &offlineEnergyRecorder;
    ObservedEnergyProbe observedEnergyProbe(numNodes,
                                            telemetryInterval,
                                            simTime - 0.1,
                                            outputPath + ".observed-energy.csv",
                                            RecoveryCoverageBound().GetSeconds());
    g_observedEnergyProbe = &observedEnergyProbe;
    ObservedEnergyProbe apObservedEnergyProbe(numNodes, telemetryInterval, simTime - 0.1,
        outputPath + ".ap-observed-energy.csv", RecoveryCoverageBound().GetSeconds());
    ApReceptionAudit apReceptionAudit(numNodes, simTime, outputPath + ".ap-reception.csv");

    InternetStackHelper internet;
    internet.Install(aps);
    internet.Install(stas);

    Ipv4AddressHelper address;
    ApplicationContainer servers;
    ApplicationContainer clients;
    const DataRate baseRate(dataRate);
    NetDeviceContainer allWifiDevices;
    std::vector<Ptr<WifiNetDevice>> apWifiDevices,staWifiDevices;
    std::vector<Ptr<OnOffApplication>> sourceApps;
    std::vector<uint64_t> policyNominalBps;
    std::ofstream burstDemand;
    if(burstScenario) {
        burstDemand.open(outputPath+".burst-demand.csv");
        NS_ABORT_MSG_IF(!burstDemand,"burst demand provenance unavailable");
        burstDemand<<"node,nominalBps,peakBps,onTimeNs,totalNominalBytes,roundingShortfallTwentiethBits\n";
        burstDemand<<std::setprecision(17);
    }

    for (uint32_t i = 0; i < numNodes; ++i)
    {
        const Ssid ssid = Ssid("mesh-dca-" + std::to_string(i));
        WifiMacHelper mac;
        if(enableSounding)
            mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid), "BeaconGeneration", BooleanValue(false));
        else
            mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
        NetDeviceContainer apDevice;
        NetDeviceContainer staDevice;
        SpectrumWifiPhyHelper phy;
        const bool retainSpectrum=compactSounding&&probeFreshSounding&&probeScanQuiescence&&enableChannelController;
        if(retainSpectrum)AddRetainedSpectrum(phy,sharedSpectrumChannel);
        else phy.SetChannel(sharedSpectrumChannel);
        phy.Set("ChannelSettings", StringValue(ChannelSettings(assignment[i])));
        apDevice = wifi.Install(phy, mac, aps.Get(i));
        mac.SetType("ns3::StaWifiMac",
                    "Ssid",
                    SsidValue(ssid),
                    "ActiveProbing",
                    BooleanValue(false));
        staDevice = wifi.Install(phy, mac, stas.Get(i));
        if(retainSpectrum)for(auto device:{apDevice.Get(0),staDevice.Get(0)}) {
            auto spectrum=DynamicCast<SpectrumWifiPhy>(DynamicCast<WifiNetDevice>(device)->GetPhy());
            Simulator::Schedule(MilliSeconds(1),[spectrum]{ConfigureRetainedSpectrum(spectrum);});
        }

        for (uint32_t role = 0; role < 2; ++role)
        {
            auto node = role == 0 ? aps.Get(i) : stas.Get(i);
            auto dev = DynamicCast<WifiNetDevice>(role == 0 ? apDevice.Get(0) : staDevice.Get(0));
            auto pos = node->GetObject<MobilityModel>()->GetPosition();
            geometryAudit << i << ',' << (role == 0 ? "AP" : "STA") << ','
                          << pos.x << ',' << pos.y << ',' << pos.z << ','
                          << static_cast<unsigned>(dev->GetPhy()->GetOperatingChannel().GetNumber()) << '\n';
            geometryAudit.flush();
            NS_ABORT_MSG_IF(!geometryAudit, "geometry audit write failed");
        }
        NetDeviceContainer pair;
        pair.Add(apDevice.Get(0));
        pair.Add(staDevice.Get(0));
        Ptr<WifiNetDevice> staWifiDevice = DynamicCast<WifiNetDevice>(staDevice.Get(0));
        Ptr<WifiNetDevice> apWifiDevice = DynamicCast<WifiNetDevice>(apDevice.Get(0));
        NS_ABORT_MSG_IF(!apWifiDevice, "expected a WifiNetDevice for AP " << i);
        NS_ABORT_MSG_IF(!staWifiDevice, "expected a WifiNetDevice for STA " << i);
        allWifiDevices.Add(apDevice.Get(0)); allWifiDevices.Add(staDevice.Get(0));
        apWifiDevices.push_back(apWifiDevice); staWifiDevices.push_back(staWifiDevice);
        Ptr<WifiRemoteStationManager> stationManager =
            staWifiDevice->GetRemoteStationManager();
        telemetryController.AddDevice(staWifiDevice);
        corrected.AddDevice(i, staWifiDevice);
        if(g_packetCohort) {
            NS_ABORT_MSG_IF(!staWifiDevice->GetMac()->TraceConnectWithoutContext("AckedMpdu",
                MakeBoundCallback(&PacketCohort::Ack,g_packetCohort,i)),"missing cohort MAC ACK trace");
            NS_ABORT_MSG_IF(!staWifiDevice->GetMac()->TraceConnectWithoutContext("DroppedMpdu",
                MakeBoundCallback(&PacketCohort::Drop,g_packetCohort,i)),"missing cohort MAC drop trace");
        }
        NS_ABORT_MSG_IF(!staWifiDevice->GetPhy()->TraceConnectWithoutContext("PhyTxPsduBegin", MakeBoundCallback(&CorrectedAttempt, i)), "missing actual attempt trace");
        NS_ABORT_MSG_IF(!staWifiDevice->GetMac()->TraceConnectWithoutContext("AckedMpdu", MakeBoundCallback(&CorrectedAck, i)), "missing MPDU acknowledgement trace");
        NS_ABORT_MSG_IF(!staWifiDevice->GetMac()->TraceConnectWithoutContext("DroppedMpdu", MakeBoundCallback(&CorrectedDrop, i)), "missing MPDU terminal drop trace");
        offlineEnergyRecorder.AddDevice(i, staWifiDevice);
        observedEnergyProbe.AddDevice(i, staWifiDevice);
        apObservedEnergyProbe.AddDevice(i, apWifiDevice);
        apReceptionAudit.Pair(i, apWifiDevice->GetMac()->GetAddress(), staWifiDevice->GetMac()->GetAddress());
        Ptr<SpectrumWifiPhy> apSpectrumPhy = DynamicCast<SpectrumWifiPhy>(apWifiDevice->GetPhy());
        NS_ABORT_MSG_IF(!apSpectrumPhy, "expected AP SpectrumWifiPhy");
        NS_ABORT_MSG_IF(!apSpectrumPhy->TraceConnectWithoutContext("ReceivedPsd",
            MakeBoundCallback(&ApObservedSignalArrival, &apObservedEnergyProbe, i)), "missing AP ReceivedPsd");
        NS_ABORT_MSG_IF(!apSpectrumPhy->GetState()->TraceConnectWithoutContext("RxOutcome",
            MakeBoundCallback(&ApReceptionAudit::Outcome, &apReceptionAudit, i)), "missing AP RxOutcome");
        NS_ABORT_MSG_IF(!apSpectrumPhy->TraceConnectWithoutContext("PhyRxDrop",
            MakeBoundCallback(&ApReceptionAudit::Drop, &apReceptionAudit, i)), "missing AP PhyRxDrop");
        telemetryController.AddManager(stationManager);
        stationManager->TraceConnectWithoutContext("MacTxDataFailed",
                                                   MakeBoundCallback(&TelemetryRetry, i));
        stationManager->TraceConnectWithoutContext("MacTxFinalDataFailed",
                                                   MakeBoundCallback(&TelemetryFinalFailure, i));
        staWifiDevice->GetMac()->TraceConnectWithoutContext(
            "MacTx",
            MakeBoundCallback(&TelemetryMacTx, i));
        staWifiDevice->GetPhy()->GetState()->TraceConnectWithoutContext(
            "RxOk",
            MakeBoundCallback(&TelemetryPhyRxOk, i));
        staWifiDevice->GetPhy()->GetState()->TraceConnectWithoutContext(
            "RxError",
            MakeBoundCallback(&TelemetryPhyRxError, i));
        staWifiDevice->GetPhy()->GetState()->TraceConnectWithoutContext(
            "State",
            MakeBoundCallback(&TelemetryPhyState, i));
        Ptr<SpectrumWifiPhy> spectrumPhy = DynamicCast<SpectrumWifiPhy>(staWifiDevice->GetPhy());
        NS_ABORT_MSG_IF(!spectrumPhy, "A5 probe requires common SpectrumWifiPhy for STA " << i);
        spectrumPhy->TraceConnectWithoutContext(
            "SignalArrival",
            MakeBoundCallback(&OfflineSignalArrival, i));
        NS_ABORT_MSG_IF(!spectrumPhy->TraceConnectWithoutContext(
            "ReceivedPsd",
            MakeBoundCallback(&ObservedSignalArrival, i)), "missing received-PSD measurement trace");
        const std::string network = "10.1." + std::to_string(i + 1) + ".0";
        address.SetBase(Ipv4Address(network.c_str()), Ipv4Mask("255.255.255.0"));
        Ipv4InterfaceContainer interfaces = address.Assign(pair);

        const uint16_t port = static_cast<uint16_t>(9000 + i);
        PacketSinkHelper sink("ns3::UdpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), port));
        ApplicationContainer installedServer = sink.Install(aps.Get(i));
        Ptr<PacketSink> packetSink = DynamicCast<PacketSink>(installedServer.Get(0));
        NS_ABORT_MSG_IF(!packetSink, "expected a PacketSink for AP " << i);
        packetSink->TraceConnectWithoutContext("Rx", MakeBoundCallback(&TelemetryAppRx, i));
        servers.Add(installedServer);

        OnOffHelper source("ns3::UdpSocketFactory",
                           InetSocketAddress(interfaces.GetAddress(0), port));
        const double load = importedLoads.empty() ? 1.0 : importedLoads[i];
        uint64_t rate = scaleRateByLoad
                            ? static_cast<uint64_t>(baseRate.GetBitRate() * std::max(load, 0.01))
                            : baseRate.GetBitRate();
        const uint64_t nominalRate=rate;
        if (burstScenario && useV31TrafficProfile)
        {
            rate = BurstPeakBps(nominalRate);
            burstDemand<<i<<','<<nominalRate<<','<<rate<<",1750000000,"
                <<BurstNominalBytes(nominalRate,1000000000,10000000000)<<','<<nominalRate*178%35<<'\n';
        }
        source.SetAttribute("DataRate", DataRateValue(DataRate(rate)));
        source.SetAttribute("PacketSize", UintegerValue(payloadSize));
        if (burstScenario)
        {
            source.SetAttribute("OnTime",
                                StringValue("ns3::ConstantRandomVariable[Constant=0.05]"));
            source.SetAttribute("OffTime",
                                StringValue("ns3::ConstantRandomVariable[Constant=0.20]"));
        }
        else if (useV31TrafficProfile)
        {
            source.SetAttribute("OnTime",
                                StringValue("ns3::ConstantRandomVariable[Constant=1000.0]"));
            source.SetAttribute("OffTime",
                                StringValue("ns3::ConstantRandomVariable[Constant=0.0]"));
        }
        ApplicationContainer installedClient = source.Install(stas.Get(i));
        Ptr<OnOffApplication> onOff = DynamicCast<OnOffApplication>(installedClient.Get(0));
        NS_ABORT_MSG_IF(!onOff, "expected an OnOffApplication for STA " << i);
        telemetryController.AddApplication(onOff, DataRate(rate),burstScenario?nominalRate:0);
        sourceApps.push_back(onOff);
        policyNominalBps.push_back(nominalRate);
        onOff->TraceConnectWithoutContext("Tx", MakeBoundCallback(&TelemetryAppTx, i));
        onOff->TraceConnectWithoutContext("OnOffState", MakeBoundCallback(&TelemetryOnOffState, i));
        clients.Add(installedClient);
    }

    std::unique_ptr<QueueContinuity> queueContinuity;
    if(enableQueueContinuity) {
        queueContinuity=std::make_unique<QueueContinuity>(outputPath+".queue-continuity.csv");
        for(uint32_t node=0;node<numNodes;++node) {
            queueContinuity->AddDevice(node,"AP",apWifiDevices[node]);
            queueContinuity->AddDevice(node,"STA",staWifiDevices[node]);
            queueContinuity->Checkpoint(node,0,"initial");
        }
    }
    InitialSounding initialSounding;
    if(enableSounding)
        initialSounding=ScheduleInitialSounding(stas,apWifiDevices,staWifiDevices,
            sharedSpectrumChannel,assignment,outputPath);
    RobustnessAudit probe(outputPath);
    for(unsigned n=0;n<staWifiDevices.size();++n) {
        auto mac=staWifiDevices[n]->GetMac();
        NS_ABORT_MSG_IF(!mac->TraceConnectWithoutContext("Assoc",MakeBoundCallback(&RobustnessAudit::Association,&probe,n,std::string("assoc"))),"Assoc trace absent");
        NS_ABORT_MSG_IF(!mac->TraceConnectWithoutContext("DeAssoc",MakeBoundCallback(&RobustnessAudit::Association,&probe,n,std::string("deassoc"))),"DeAssoc trace absent");
        NS_ABORT_MSG_IF(!mac->TraceConnectWithoutContext("BeaconArrival",MakeBoundCallback(&RobustnessAudit::Beacon,&probe,n)),"BeaconArrival trace absent");
        for(auto role:{std::string("STA"),std::string("AP")}) {
            auto device=role=="STA"?staWifiDevices[n]:apWifiDevices[n];
            NS_ABORT_MSG_IF(!device->GetPhy()->TraceConnectWithoutContext("SignalArrival",MakeBoundCallback(&RobustnessAudit::Emitter,&probe,role,n)),"emitter trace absent");
        }
    }
    for(unsigned n=0;n<2;++n)probe.addresses.push_back(staWifiDevices[n]->GetMac()->GetAddress());
    for(unsigned n=0;n<2;++n) {
        auto phy=DynamicCast<SpectrumWifiPhy>(staWifiDevices[n]->GetPhy());
        NS_ABORT_MSG_IF(!phy->GetState()->TraceConnectWithoutContext("RxOk",MakeBoundCallback(&RobustnessAudit::Heard,&probe,n)),"probe RxOk absent");
        NS_ABORT_MSG_IF(!phy->TraceConnectWithoutContext("PhyRxBegin",MakeBoundCallback(&RobustnessAudit::Begin,&probe,n)),"probe RxBegin absent");
        for(const auto* name:{"RxSensitivity","CcaEdThreshold","CcaSensitivity"}) {
            DoubleValue value;
            if(phy->GetAttributeFailSafe(name,value))probe.attributes<<n<<','<<name<<','<<value.Get()<<'\n';
        }
        for(auto role:{std::string("STA"),std::string("AP")}) {
            auto d=role=="STA"?staWifiDevices[n]:apWifiDevices[n];
            auto spectrum=DynamicCast<SpectrumWifiPhy>(d->GetPhy());
            NS_ABORT_MSG_IF(!spectrum->TraceConnectWithoutContext("SignalArrival",MakeBoundCallback(&RobustnessAudit::Arrival,&probe,role,n)),"probe arrival absent");
        }
    }
    for(double at:{1.0,3.0,5.0,7.0,9.5})Simulator::Schedule(Seconds(at),[&probe,stas,staWifiDevices]{probe.Sample(stas,staWifiDevices);});
    ValidateRecoveryDomain(allWifiDevices,adjacentWaveform,propagationDelay);
    std::unique_ptr<ChannelController> channelController;
    std::unique_ptr<FreshSounding> freshSounding;
    if(enableChannelController)
    {
        channelController=std::make_unique<ChannelController>(apWifiDevices,staWifiDevices,
            allWifiDevices,adjacentWaveform,propagationDelay,corrected,
            telemetryController,sourceApps,telemetryInterval,simTime,
            channelControlSchedule,outputPath+".channel-control.csv",queueContinuity.get(),trackingMode=="window");
        channelController->Start();
        if(probeFreshSounding) {
            freshSounding=std::make_unique<FreshSounding>(stas,apWifiDevices,sharedSpectrumChannel,outputPath,compactSounding);
            channelController->SetFreshSounding(freshSounding.get());
        }
    }
    std::unique_ptr<threeway::PolicyBridge> policyBridge;
    std::unique_ptr<DemandSchedule> demandSchedule;
    std::ofstream demandChanges;
    if(!demandScheduleFile.empty()) {
        demandSchedule=std::make_unique<DemandSchedule>(numNodes,simTime,demandScheduleFile);
        demandChanges.open(outputPath+".demand-changes.csv");NS_ABORT_MSG_IF(!demandChanges,"cannot record demand changes");
        demandChanges<<"changeAtNs,node,oldBps,newBps,sourceWasPaused\n";
        auto currentRates=std::make_shared<std::vector<uint64_t>>(policyNominalBps);
        demandSchedule->Start([&telemetryController,&sourceApps,&demandChanges,&queueContinuity,currentRates](uint32_t n,uint64_t bps){
            demandChanges<<Simulator::Now().GetNanoSeconds()<<','<<n<<','<<currentRates->at(n)<<','<<bps<<','
                <<sourceApps[n]->IsChannelSwitchPaused()<<'\n';
            const auto at=Simulator::Now().GetNanoSeconds();
            if(queueContinuity)queueContinuity->Checkpoint(n,at,"before_rate");
            currentRates->at(n)=bps;telemetryController.SetProspectiveRate(n,bps);
            if(queueContinuity)queueContinuity->Checkpoint(n,at,"after_rate");
        });
    }
    if(enablePolicyBridge) {
        policyBridge=std::make_unique<threeway::PolicyBridge>(aps,
            [&channelController](uint32_t n){return channelController->ReadChannel(n);},
            [&channelController](uint32_t n,uint8_t c,uint64_t epoch,threeway::ManagementEpochs::Done done){
                channelController->Execute(n,c,epoch,std::move(done));
            },*initialSounding.receiver,corrected,policyNominalBps,telemetryInterval,simTime,outputPath,
            managementDropAckEpoch,managementDropAckNode,demandSchedule.get(),freshSounding.get());
        if(freshSounding)freshSounding->reportReady=[&policyBridge](uint64_t epoch){policyBridge->FreshReady(epoch);};
        if(freshSounding)freshSounding->reportUnavailable=[&policyBridge](uint64_t epoch){policyBridge->FreshUnavailable(epoch);};
        policyBridge->SetActionSettled([&channelController](uint64_t epoch){channelController->TraceTracking(epoch,"action_settled",true);channelController->CloseTracking(epoch);});
        policyBridge->Start();
    }

    servers.Start(Seconds(0.5));
    servers.Stop(Seconds(simTime));
    clients.Start(Seconds(1.0));
    clients.Stop(Seconds(simTime - 0.1));
    std::ofstream scanReadiness;
    if(probeScanQuiescence) {
        scanReadiness.open(outputPath+".scan-readiness.csv");NS_ABORT_MSG_IF(!scanReadiness,"scan readiness trace unavailable");
        scanReadiness<<"epoch,pauseAtNs,terminalAtNs,ready,nonemptyPairs,busyPairs,responsePairs,guardPairs\n";
        for(unsigned epoch=0;epoch<13;++epoch) {
            const auto at=Seconds(1)+MicroSeconds(1)+epoch*Seconds(telemetryInterval);
            Simulator::Schedule(at-RecoveryCoverageBound(),[&channelController,epoch]{channelController->OpenTracking(epoch+1);});
            Simulator::Schedule(Seconds(1)+MicroSeconds(1)+epoch*Seconds(telemetryInterval),
                [&channelController,&scanReadiness,epoch]{channelController->ProbeScanQuiescence(epoch+1,scanReadiness);});
        }
    }
    Simulator::Schedule(Seconds(simTime - 0.1),
                        &TelemetryRiskController::StopDemand,
                        &telemetryController);

    FlowMonitorHelper flowHelper;
    Ptr<FlowMonitor> monitor = flowHelper.InstallAll();

    Simulator::Schedule(Seconds(1.0 + telemetryInterval),
                        &TelemetryRiskController::Sample,
                        &telemetryController);
    observedEnergyProbe.Start(Seconds(1.0));
    apObservedEnergyProbe.Start(Seconds(1.0));
    corrected.Start();
    if (adjacentExcitationScenario && enableAdjacentExcitation)
    {
        Simulator::Schedule(Seconds(1.0), &WaveformGenerator::Start, adjacentWaveform);
        Simulator::Schedule(Seconds(simTime - 0.1), &WaveformGenerator::Stop, adjacentWaveform);
    }

    Simulator::Stop(developmentStopAtNs?NanoSeconds(developmentStopAtNs):Seconds(simTime));
    Simulator::Run();
    if(queueContinuity)queueContinuity->Finish();
    if(developmentStopAtNs) {
        std::cout<<"THREEWAY_DIAGNOSTIC_STOP_NO_SERVICE_SCORE "<<Simulator::Now().GetNanoSeconds()<<std::endl;
        Simulator::Destroy();return 0;
    }
    corrected.Write();
    if(packetCohort)packetCohort->Write();
    if(demandSchedule){demandSchedule->Write(telemetryInterval,outputPath+".demand-windows.csv");demandChanges.flush();}
    telemetryController.Finalize();
    observedEnergyProbe.Finalize();
    apObservedEnergyProbe.Finalize();
    apReceptionAudit.Finish();
    monitor->CheckForLostPackets();
    offlineEnergyRecorder.Write();

    std::ofstream output(outputPath);
    output << "flowId,source,destination,txPackets,rxPackets,lossRate,throughputMbps,meanDelayMs\n";
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowHelper.GetClassifier());
    auto stats = monitor->GetFlowStats();
    for (const auto& entry : stats)
    {
        const auto tuple = classifier->FindFlow(entry.first);
        const auto& flow = entry.second;
        const double duration = (flow.timeLastRxPacket - flow.timeFirstTxPacket).GetSeconds();
        const double throughputMbps = duration > 0.0
                                          ? (flow.rxBytes * 8.0 / duration / 1e6)
                                          : 0.0;
        const double meanDelayMs = flow.rxPackets > 0
                                       ? (flow.delaySum.GetSeconds() * 1000.0 / flow.rxPackets)
                                       : 0.0;
        const double lossRate = flow.txPackets > 0
                                    ? (1.0 - static_cast<double>(flow.rxPackets) / flow.txPackets)
                                    : 0.0;
        output << entry.first << ',' << tuple.sourceAddress << ',' << tuple.destinationAddress << ','
               << flow.txPackets << ',' << flow.rxPackets << ',' << lossRate << ','
               << throughputMbps << ',' << meanDelayMs << '\n';
    }
    output.close();

    monitor->SerializeToXmlFile(flowMonitorOutput, true, true);
    Simulator::Destroy();
    return 0;
}
