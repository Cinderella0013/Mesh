import unittest
import numpy as np
from q_margins import decision_margins

CFG=dict(pref=1.,eta=0.,lam=1.,mu=.02,delta=.01)

class DecisionMarginChecks(unittest.TestCase):
    def test_q_alone_cannot_cross_switch_penalty_and_hysteresis(self):
        rows=decision_margins(np.array([[0.,1.]]),np.array([0]),np.zeros((1,1)),np.array([0]),np.zeros((1,1),bool),np.array([True]),CFG,.02,[0])
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['local_zero_q_selected'],0)
        self.assertFalse(rows[0]['local_q_changes_final_choice'])
        self.assertEqual(rows[0]['recorded_selected'],0)

    def test_q_can_bind_between_alternatives_without_overcoming_stay_cost_alone(self):
        rows=decision_margins(np.array([[0.,1.,0.],[0.,0.,0.],[0.,0.,0.]]),np.array([0,0,1]),np.array([[0.,.05,.01],[0.,0.,0.],[0.,0.,0.]]),np.array([0,1,2]),np.zeros((3,3),bool),np.ones(3,bool),CFG,.02,[1,0,1])
        row=next(r for r in rows if r['node']==0)
        self.assertEqual(row['local_zero_q_selected'],2)
        self.assertTrue(row['local_q_changes_final_choice'])
        self.assertEqual(row['epoch_zero_q_selected'],2)
        self.assertGreater(row['conditional_switch_margin'],0)

    def test_gate_can_override_a_disallowed_current_channel(self):
        rows=decision_margins(np.zeros((2,2)),np.array([0,0]),np.zeros((2,2)),np.array([0,1]),np.array([[False,True],[True,False]]),np.array([True,False]),CFG,.02,[0,0])
        row=next(r for r in rows if r['node']==1)
        self.assertFalse(row['current_allowed'])
        self.assertTrue(row['gate_blocked'])
        self.assertFalse(row['local_q_changes_final_choice'])
        self.assertIsNone(row['conditional_switch_margin'])

    def test_recorded_action_disagreement_is_not_silently_analyzed(self):
        with self.assertRaisesRegex(AssertionError,'recorded action'):
            decision_margins(np.array([[0.,1.]]),np.array([0]),np.zeros((1,1)),np.array([0]),np.zeros((1,1),bool),np.array([True]),CFG,.02,[1])

if __name__=='__main__':unittest.main(verbosity=2)
