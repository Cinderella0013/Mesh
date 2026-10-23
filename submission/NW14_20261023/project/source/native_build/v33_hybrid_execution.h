#pragma once

#include "v33_hybrid_state.h"

#include <cstddef>
#include <iomanip>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace v33 {

inline void ConfigureTelemetryStream(std::ostream& output) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
}

enum class HybridExecutionMode { Baseline, TelemetryRts, CapacityPacing, Hybrid };

struct HybridTelemetryAction {
    RiskClass riskClass;
    bool rtsEnabled;
    bool guardChanged;
    double elapsedPacingLevel;
    double nextPacingLevel;
    bool pacingChanged;
};

class HybridExecutionController {
  public:
    HybridExecutionController(std::size_t nodeCount,
                              HybridConfig config,
                              HybridExecutionMode mode,
                              bool initialRtsEnabled = false)
        : m_mode(mode),
          m_globalActivationWindows(config.activationWindows),
          m_appliedRts(nodeCount, initialRtsEnabled),
          m_appliedPacing(nodeCount, 1.00) {
        if (nodeCount == 0) {
            throw std::invalid_argument("nodeCount must be positive");
        }
        m_states.reserve(nodeCount);
        for (std::size_t index = 0; index < nodeCount; ++index) {
            m_states.emplace_back(config);
        }
    }

    std::vector<HybridTelemetryAction>
    UpdateWindow(const std::vector<HybridInput>& inputs) {
        if (inputs.size() != m_states.size()) {
            throw std::invalid_argument("input count must match nodeCount");
        }

        const bool globalCapacityActive = inputs.front().globalCapacityActive;
        for (const auto& input : inputs) {
            if (input.globalCapacityActive != globalCapacityActive) {
                throw std::invalid_argument(
                    "globalCapacityActive must match for every node");
            }
        }

        std::vector<HybridAction> actions;
        actions.reserve(m_states.size());
        for (std::size_t index = 0; index < m_states.size(); ++index) {
            actions.push_back(m_states[index].Update(inputs[index]));
        }

        if (globalCapacityActive) {
            ++m_globalCapacityWindows;
        } else {
            m_globalCapacityWindows = 0;
        }
        if (globalCapacityActive &&
            m_globalCapacityWindows >= m_globalActivationWindows) {
            SynchronizeGlobalCapacity(m_states, actions);
        }

        const bool rtsEnabledByMode =
            m_mode == HybridExecutionMode::TelemetryRts ||
            m_mode == HybridExecutionMode::Hybrid;
        const bool pacingEnabledByMode =
            m_mode == HybridExecutionMode::CapacityPacing ||
            m_mode == HybridExecutionMode::Hybrid;

        std::vector<HybridTelemetryAction> telemetry;
        telemetry.reserve(actions.size());
        for (std::size_t index = 0; index < actions.size(); ++index) {
            const bool appliedRts = rtsEnabledByMode
                                        ? actions[index].rtsEnabled
                                        : m_appliedRts[index];
            const double elapsedPacing = m_appliedPacing[index];
            const double nextPacing =
                pacingEnabledByMode ? actions[index].pacingFactor : 1.00;
            telemetry.push_back({
                actions[index].riskClass,
                appliedRts,
                appliedRts != m_appliedRts[index],
                elapsedPacing,
                nextPacing,
                nextPacing != elapsedPacing,
            });
            m_appliedRts[index] = appliedRts;
            m_appliedPacing[index] = nextPacing;
        }
        return telemetry;
    }

  private:
    std::vector<HybridActuatorState> m_states;
    HybridExecutionMode m_mode;
    unsigned m_globalActivationWindows;
    unsigned m_globalCapacityWindows{0};
    std::vector<bool> m_appliedRts;
    std::vector<double> m_appliedPacing;
};

} // namespace v33
