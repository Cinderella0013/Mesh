#pragma once
#include "ns3/core-module.h"
#include "ns3/mobility-module.h"
#include "ns3/node-list.h"
#include "ns3/spectrum-module.h"
#include "ns3/wifi-module.h"
#include <cmath>
#include <set>
inline ns3::Time RecoveryPropagationBound()
{
 using namespace ns3;
 const auto propagationNs=static_cast<int64_t>(std::ceil(std::sqrt(150.0*150.0+150.0*150.0+1.5*1.5)/299792458.0*1e9));
 return NanoSeconds(propagationNs);
}
inline ns3::Time RecoveryCoverageBound()
{
 using namespace ns3;
 return MilliSeconds(100)+RecoveryPropagationBound();
}
// Call before simulation and at each channel actuation. Launcher identity gate
// must succeed separately; this guard alone does not authorize a bound.
inline void ValidateRecoveryDomain(const ns3::NetDeviceContainer& wifi,
                                   ns3::Ptr<ns3::WaveformGenerator> waveform,
                                   ns3::Ptr<ns3::ConstantSpeedPropagationDelayModel> delay)
{
 using namespace ns3;
 NS_ABORT_MSG_IF(Time::GetResolution()!=Time::NS || wifi.GetN()!=30 || NodeList::GetNNodes()!=31, "recovery certificate time/transmitter domain");
 NS_ABORT_MSG_IF(!delay || !waveform,"recovery certificate missing component");
 auto emitter=waveform->GetMobility();
 NS_ABORT_MSG_IF(!emitter,"recovery certificate missing emitter mobility");
 auto e=emitter->GetPosition();
 NS_ABORT_MSG_IF(!std::isfinite(e.x)||!std::isfinite(e.y)||!std::isfinite(e.z)||e.x<0||e.x>150||e.y<0||e.y>150||e.z<0||e.z>1.5,"recovery certificate emitter geometry");
 DoubleValue speed; delay->GetAttribute("Speed",speed);
 NS_ABORT_MSG_IF(speed.Get()!=299792458 || !waveform || waveform->GetPeriod()!=MilliSeconds(100) || waveform->GetDutyCycle()!=1, "recovery certificate waveform/propagation domain");
 NS_ABORT_MSG_IF(RecoveryCoverageBound()!=NanoSeconds(100000708), "recovery certificate arithmetic domain");
 std::set<Ptr<NetDevice>> unique;
 for(unsigned i=0;i<wifi.GetN();++i) {
   NS_ABORT_MSG_IF(!unique.insert(wifi.Get(i)).second,"recovery certificate duplicate transmitter");
   auto d=DynamicCast<WifiNetDevice>(wifi.Get(i));
   NS_ABORT_MSG_IF(!d,"recovery certificate transmitter type");
   auto p=d->GetPhy(); unsigned c=p->GetChannelNumber();
   NS_ABORT_MSG_IF(p->GetStandard()!=WIFI_STANDARD_80211a || p->GetChannelWidth()!=MHz_u{20} || p->GetPhyBand()!=WIFI_PHY_BAND_5GHZ || !(c==36||c==40||c==44||c==48),"recovery certificate PHY domain");
   auto mobility=d->GetNode()->GetObject<MobilityModel>();
   NS_ABORT_MSG_IF(!mobility,"recovery certificate missing mobility");
   auto pos=mobility->GetPosition();
   NS_ABORT_MSG_IF(!std::isfinite(pos.x)||!std::isfinite(pos.y)||!std::isfinite(pos.z)||pos.x<0||pos.x>150||pos.y<0||pos.y>150||pos.z<0||pos.z>1.5,"recovery certificate geometry");
 }
}
