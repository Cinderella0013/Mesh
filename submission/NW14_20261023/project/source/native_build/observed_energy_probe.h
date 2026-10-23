#pragma once
#include "observed_energy_window.h"
#include "ns3/wifi-module.h"
#include "ns3/spectrum-module.h"
#include <fstream>
#include <iomanip>
#include <limits>
using namespace ns3;
class ObservedEnergyProbe
{
  public:
    ObservedEnergyProbe(uint32_t numNodes,
                        double intervalSeconds,
                        double stopTime,
                        const std::string& outputPath,
                        double certifiedCoverageBoundSeconds = -1)
        : m_totals(numNodes),
          m_devices(numNodes),
          m_initialChannels(numNodes),
          m_initialSupported(numNodes, false),
          m_certificateBrokenAt(numNodes, std::numeric_limits<double>::infinity()),
          m_coverageBound(certifiedCoverageBoundSeconds),
          m_eventsOutput(outputPath + ".events.csv"),
          m_intervalSeconds(intervalSeconds),
          m_stopTime(stopTime),
          m_output(outputPath)
    {
        NS_ABORT_MSG_IF(!m_output, "cannot open observed energy output: " << outputPath);
        NS_ABORT_MSG_IF(!m_eventsOutput, "cannot open switch-event output");
        NS_ABORT_MSG_IF(!std::isfinite(m_coverageBound) || (m_coverageBound < 0 && m_coverageBound != -1), "invalid certified coverage bound");
        m_output << std::setprecision(17);
        m_eventsOutput << std::setprecision(17);
        m_output << "observedAt,node,channel,wantedRawEnergyJ,adjacentRawEnergyJ,observationValid,reason,windowStartS,windowEndS,channelEpoch,channelChanges,channelEventsValid\n";
        m_eventsOutput << "timeS,node,oldChannel,newChannel,channelEpoch,switchingEndS\n";
    }

    void AddDevice(uint32_t node, Ptr<WifiNetDevice> device)
    {
        m_devices.at(node) = device;
        NS_ABORT_MSG_IF(!device, "missing observed-energy device");
        m_initialChannels.at(node) = device->GetPhy()->GetChannelNumber();
        m_initialSupported.at(node) = Supported(device);
        NS_ABORT_MSG_IF(!device->GetPhy()->GetState()->TraceConnectWithoutContext(
            "State", MakeBoundCallback(&ObservedEnergyProbe::PhyState, this, node)), "missing PHY state trace");
    }

    void SignalArrival(uint32_t node,
                       Ptr<const SpectrumValue> psd,
                       Time duration)
    {
        const double arrival=Simulator::Now().GetSeconds();
        if (!std::isfinite(duration.GetSeconds()) || duration.GetSeconds() <= 0)
        {
            m_invalidArrivals.push_back({node, arrival});
            return;
        }
        if (m_coverageBound >= 0 && duration.GetSeconds() > m_coverageBound)
            m_certificateBrokenAt.at(node) = std::min(m_certificateBrokenAt.at(node), arrival);
        if (!psd)
        {
            m_totals.at(node).Add(Simulator::Now().GetSeconds(), Simulator::Now().GetSeconds()+duration.GetSeconds(), 0, 0, false);
            return;
        }
        const auto device = m_devices.at(node);
        NS_ABORT_MSG_IF(!device, "missing observed-energy device for STA " << node);
        const double centerHz = (5000.0 + 5.0 * device->GetPhy()->GetChannelNumber()) * 1e6;
        const double wantedLowHz = centerHz - 10e6;
        const double wantedHighHz = centerHz + 10e6;
        const double adjacentLowHz = centerHz - 30e6;
        const double adjacentHighHz = centerHz + 30e6;
        double wantedPower=0, adjacentPower=0;
        bool valid=Supported(device) && psd->ConstBandsBegin()!=psd->ConstBandsEnd();
        auto band = psd->ConstBandsBegin();
        auto value = psd->ConstValuesBegin();
        for (; band != psd->ConstBandsEnd(); ++band, ++value)
        {
            const double wattsPerHz = *value;
            if (!std::isfinite(wattsPerHz) || wattsPerHz < 0.0 ||
                !std::isfinite(band->fl) || !std::isfinite(band->fh) || band->fh <= band->fl)
            {
                valid=false;
                continue;
            }
            wantedPower +=
                wattsPerHz * Overlap(band->fl, band->fh, wantedLowHz, wantedHighHz);
            adjacentPower +=
                wattsPerHz * (Overlap(band->fl, band->fh, adjacentLowHz, wantedLowHz) +
                              Overlap(band->fl, band->fh, wantedHighHz, adjacentHighHz));
        }
        m_totals.at(node).Add(Simulator::Now().GetSeconds(), Simulator::Now().GetSeconds()+duration.GetSeconds(), wantedPower, adjacentPower, valid);
    }

    void Finalize()
    {
        if (Simulator::Now().GetSeconds()>m_windowStart) Sample();
    }

    void Start(Time windowStart)
    {
        m_windowStart=windowStart.GetSeconds();
        Simulator::Schedule(windowStart + Seconds(m_intervalSeconds), &ObservedEnergyProbe::Sample, this);
    }

  private:
    struct ChannelEvent { uint32_t node; double start, end; unsigned channel; uint64_t epoch; bool supported, channelChanged; };
    static bool Supported(Ptr<WifiNetDevice> device)
    {
        if (!device) return false;
        auto phy=device->GetPhy();
        auto channel=phy->GetChannelNumber();
        return phy->GetPhyBand()==WIFI_PHY_BAND_5GHZ && phy->GetChannelWidth()==MHz_u{20} &&
            (channel==36 || channel==40 || channel==44 || channel==48);
    }
    // The bound is a caller-certified environment-wide signal lifetime plus propagation delay.
    // Default -1 cannot certify unseen old-grid signals: switching remains unknown.
    static void PhyState(ObservedEnergyProbe* self, uint32_t node, Time start, Time duration, WifiPhyState state)
    {
        if (state != WifiPhyState::SWITCHING) return;
        unsigned oldChannel = self->m_initialChannels.at(node);
        uint64_t epoch = 0;
        for (const auto& event : self->m_events)
            if (event.node == node) { oldChannel = event.channel; epoch = event.epoch; }
        unsigned channel = self->m_devices.at(node)->GetPhy()->GetChannelNumber();
        self->m_events.push_back({node, start.GetSeconds(), (start + duration).GetSeconds(), channel, epoch + 1, Supported(self->m_devices.at(node)), oldChannel != channel});
        self->m_eventsOutput << start.GetSeconds() << ',' << node << ',' << oldChannel << ',' << channel << ',' << epoch + 1 << ',' << (start + duration).GetSeconds() << '\n';
        self->m_eventsOutput.flush();
    }

    static double Overlap(double lowA, double highA, double lowB, double highB)
    {
        return std::max(0.0, std::min(highA, highB) - std::max(lowA, lowB));
    }

    void Sample()
    {
        const double now = Simulator::Now().GetSeconds();
        for (uint32_t node = 0; node < m_totals.size(); ++node)
        {
            const auto total=m_totals.at(node).Take(m_windowStart,now);
            bool valid=total.valid && bool(m_devices.at(node));
            std::string reason=valid ? "configured_common_spectrum_sensor" : "invalid_or_missing_observation";
            unsigned channel=m_initialChannels.at(node);
            bool supported=m_initialSupported.at(node);
            uint64_t epoch=0, changes=0;
            // Use half-open event history, never the sample-time PHY label. Events at
            // exactly now belong to the next window regardless of simulator callback order.
            for (const auto& event : m_events)
            {
                if (event.node != node || event.start >= now) continue;
                channel=event.channel; epoch=event.epoch; supported=event.supported;
                if (event.start >= m_windowStart && event.channelChanged) ++changes;
                const bool recovering=m_coverageBound < 0 ||
                    Seconds(m_windowStart) < std::max(Seconds(event.end), Seconds(event.start) + Seconds(m_coverageBound));
                if (recovering)
                {
                    valid=false;
                    reason=m_coverageBound < 0 ? "coverage_uncertified" : "channel_transition_coverage";
                }
            }
            if (!supported) { valid=false; reason="unsupported_or_missing_channel_configuration"; }
            for (const auto& arrival : m_invalidArrivals)
                if (arrival.first==node && arrival.second>=m_windowStart && arrival.second<now)
                { valid=false; reason="invalid_arrival_duration"; }
            if (m_certificateBrokenAt.at(node)<now) { valid=false; reason="coverage_certificate_violated"; }
            const double unknown=std::numeric_limits<double>::quiet_NaN();
            m_output << now << ',' << node << ',' << channel << ','
                     << (valid ? total.wanted : unknown) << ',' << (valid ? total.adjacent : unknown) << ','
                     << (valid ? "true" : "false") << ',' << reason << ',' << m_windowStart << ',' << now << ','
                     << epoch << ',' << changes << (m_devices.at(node) ? ",true\n" : ",false\n");
        }
        m_output.flush();
        m_windowStart=now;
        if (now + m_intervalSeconds <= m_stopTime + 1e-9)
        {
            Simulator::Schedule(Seconds(m_intervalSeconds), &ObservedEnergyProbe::Sample, this);
        }
    }

    std::vector<EnergyWindow> m_totals;
    double m_windowStart=1.0;
    std::vector<Ptr<WifiNetDevice>> m_devices;
    std::vector<unsigned> m_initialChannels;
    std::vector<bool> m_initialSupported;
    std::vector<double> m_certificateBrokenAt;
    std::vector<std::pair<uint32_t, double>> m_invalidArrivals;
    double m_coverageBound;
    std::ofstream m_eventsOutput;
    std::vector<ChannelEvent> m_events;
    double m_intervalSeconds;
    double m_stopTime;
    std::ofstream m_output;
};
