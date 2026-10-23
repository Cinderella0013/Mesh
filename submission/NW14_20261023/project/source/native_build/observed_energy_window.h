#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
struct EnergyWindow {
 struct Segment { double start,end,wanted,adjacent; bool valid; };
 struct Result { double wanted=0,adjacent=0; bool valid=true; };
 std::vector<Segment> pending;
 double previousEnd=-1;
 bool invalidUnknown=false;
 void Add(double start,double end,double wanted,double adjacent,bool valid) {
  if(!std::isfinite(start)||!std::isfinite(end)||end<=start) { invalidUnknown=true; return; }
  valid=valid && std::isfinite(wanted)&&std::isfinite(adjacent)&&wanted>=0&&adjacent>=0;
  pending.push_back({start,end,valid?wanted:0,valid?adjacent:0,valid});
 }
 Result Take(double start,double end) {
  if(!std::isfinite(start)||!std::isfinite(end)||end<=start || (previousEnd>=0 && std::abs(start-previousEnd)>1e-9))
   throw std::runtime_error("invalid energy window sequence");
  Result r; r.valid=!invalidUnknown; invalidUnknown=false;
  for(const auto& s:pending) {
   double dt=std::max(0.,std::min(end,s.end)-std::max(start,s.start));
   if(dt>0) { r.wanted+=s.wanted*dt; r.adjacent+=s.adjacent*dt; r.valid=r.valid&&s.valid; }
  }
  pending.erase(std::remove_if(pending.begin(),pending.end(),[end](const auto& s){return s.end<=end;}),pending.end());
  r.valid=r.valid && std::isfinite(r.wanted) && std::isfinite(r.adjacent);
  previousEnd=end; return r;
 }
};
