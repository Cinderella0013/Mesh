#pragma once

#include <vector>

namespace v33 {

enum class RiskClass { Normal, Collision, Capacity };

struct HybridConfig {
    unsigned activationWindows{2};
    unsigned dwellWindows{2};
    double collisionReleaseRatio{0.50};
    double capacityReleaseRatio{0.50};
    double pacingFloor{0.70};
};

struct HybridInput {
    bool collisionActive;
    bool localCapacityActive;
    bool globalCapacityActive;
    double zCollision;
    double zCapacity;
};

struct HybridAction {
    RiskClass riskClass;
    bool rtsEnabled;
    double pacingFactor;
    bool rtsChanged;
    bool pacingChanged;
};

class HybridActuatorState {
  public:
    explicit HybridActuatorState(HybridConfig config = {})
        : m_config(config) {
        if (m_config.pacingFloor <= 0.70) {
            m_config.pacingFloor = 0.70;
        } else if (m_config.pacingFloor <= 0.85) {
            m_config.pacingFloor = 0.85;
        } else {
            m_config.pacingFloor = 1.00;
        }
    }

    HybridAction Update(const HybridInput& input) {
        const bool capacityActive =
            input.localCapacityActive || input.globalCapacityActive;

        ++m_dwellWindows;
        UpdateActivationWindows(input.collisionActive, capacityActive);

        switch (m_riskClass) {
        case RiskClass::Normal:
            if (CanTransition(m_capacityActiveWindows)) {
                EnterCapacity();
            } else if (CanTransition(m_collisionActiveWindows)) {
                EnterCollision();
            }
            break;
        case RiskClass::Collision:
            if (CanTransition(m_capacityActiveWindows)) {
                EnterCapacity();
            } else {
                UpdateCollisionRecovery(input);
            }
            break;
        case RiskClass::Capacity:
            UpdateCapacity(input, capacityActive);
            break;
        }

        UpdateRts(input, capacityActive);
        const HybridAction action{
            m_riskClass,
            m_rtsEnabled,
            m_pacingFactor,
            m_rtsEnabled != m_lastRtsEnabled,
            m_pacingFactor != m_lastPacingFactor,
        };
        m_lastRtsEnabled = m_rtsEnabled;
        m_lastPacingFactor = m_pacingFactor;
        return action;
    }

    void SynchronizeCapacityPacing(double pacingFactor) {
        m_riskClass = RiskClass::Capacity;
        if (pacingFactor <= 0.70 && m_config.pacingFloor <= 0.70) {
            m_pacingFactor = 0.70;
        } else if (pacingFactor <= 0.85 && m_config.pacingFloor <= 0.85) {
            m_pacingFactor = 0.85;
        } else {
            m_pacingFactor = 1.00;
        }
        m_rtsEnabled = false;
        m_lastRtsEnabled = false;
        m_lastPacingFactor = m_pacingFactor;
        ResetTransitionWindows();
    }

  private:
    bool CanTransition(unsigned consecutiveWindows) const {
        return consecutiveWindows >= m_config.activationWindows &&
               m_dwellWindows >= m_config.dwellWindows;
    }

    void UpdateActivationWindows(bool collisionActive, bool capacityActive) {
        if (capacityActive) {
            ++m_capacityActiveWindows;
            m_collisionActiveWindows = 0;
        } else if (collisionActive) {
            ++m_collisionActiveWindows;
            m_capacityActiveWindows = 0;
        } else {
            m_collisionActiveWindows = 0;
            m_capacityActiveWindows = 0;
        }
    }

    void UpdateCollisionRecovery(const HybridInput& input) {
        if (!input.collisionActive &&
            input.zCollision < m_config.collisionReleaseRatio) {
            ++m_collisionReleaseWindows;
        } else {
            m_collisionReleaseWindows = 0;
        }

        if (CanTransition(m_collisionReleaseWindows)) {
            m_riskClass = RiskClass::Normal;
            ResetTransitionWindows();
        }
    }

    void UpdateRts(const HybridInput& input, bool capacityActive) {
        if (m_riskClass != RiskClass::Collision) {
            m_rtsEnabled = false;
            return;
        }

        if (m_rtsEnabled) {
            const bool collisionReleased =
                !input.collisionActive &&
                input.zCollision < m_config.collisionReleaseRatio;
            if (capacityActive || collisionReleased) {
                m_rtsEnabled = false;
            }
            return;
        }

        if (!capacityActive && CanTransition(m_collisionActiveWindows)) {
            m_rtsEnabled = true;
            ResetTransitionWindows();
        }
    }

    void UpdateCapacity(const HybridInput& input, bool capacityActive) {
        if (capacityActive) {
            m_capacityReleaseWindows = 0;
            if (m_pacingFactor > m_config.pacingFloor &&
                CanTransition(m_capacityActiveWindows)) {
                m_pacingFactor = 0.70;
                ResetTransitionWindows();
            }
            return;
        }

        if (input.zCapacity < m_config.capacityReleaseRatio) {
            ++m_capacityReleaseWindows;
        } else {
            m_capacityReleaseWindows = 0;
        }

        if (!CanTransition(m_capacityReleaseWindows)) {
            return;
        }

        if (m_pacingFactor < 0.85) {
            m_pacingFactor = 0.85;
        } else {
            m_pacingFactor = 1.00;
            m_riskClass = RiskClass::Normal;
        }
        ResetTransitionWindows();
    }

    void EnterCollision() {
        m_riskClass = RiskClass::Collision;
        m_rtsEnabled = true;
        ResetTransitionWindows();
    }

    void EnterCapacity() {
        m_riskClass = RiskClass::Capacity;
        m_rtsEnabled = false;
        m_pacingFactor = m_config.pacingFloor > 0.85 ? 1.00 : 0.85;
        ResetTransitionWindows();
    }

    void ResetTransitionWindows() {
        m_collisionActiveWindows = 0;
        m_capacityActiveWindows = 0;
        m_collisionReleaseWindows = 0;
        m_capacityReleaseWindows = 0;
        m_dwellWindows = 0;
    }

    HybridConfig m_config;
    RiskClass m_riskClass{RiskClass::Normal};
    double m_pacingFactor{1.00};
    bool m_rtsEnabled{false};
    bool m_lastRtsEnabled{false};
    double m_lastPacingFactor{1.00};
    unsigned m_collisionActiveWindows{0};
    unsigned m_capacityActiveWindows{0};
    unsigned m_collisionReleaseWindows{0};
    unsigned m_capacityReleaseWindows{0};
    unsigned m_dwellWindows{0};
};

inline bool SynchronizeGlobalCapacity(std::vector<HybridActuatorState>& states,
                                      std::vector<HybridAction>& actions) {
    if (states.empty() || states.size() != actions.size()) {
        return false;
    }

    double commonFactor = actions.front().pacingFactor;
    for (const auto& action : actions) {
        if (action.pacingFactor < commonFactor) {
            commonFactor = action.pacingFactor;
        }
    }

    bool factorsDiffer = false;
    for (const auto& action : actions) {
        factorsDiffer = factorsDiffer || action.pacingFactor != commonFactor;
    }
    if (!factorsDiffer) {
        return false;
    }

    for (std::size_t index = 0; index < actions.size(); ++index) {
        auto& action = actions[index];
        const bool rtsWasEnabled = action.rtsEnabled;
        const bool pacingWasDifferent = action.pacingFactor != commonFactor;
        states[index].SynchronizeCapacityPacing(commonFactor);
        action.riskClass = RiskClass::Capacity;
        action.rtsEnabled = false;
        action.pacingFactor = commonFactor;
        action.rtsChanged = action.rtsChanged || rtsWasEnabled;
        action.pacingChanged = action.pacingChanged || pacingWasDifferent;
    }
    return true;
}

} // namespace v33
