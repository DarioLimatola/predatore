import copy
import tempfile
from pathlib import Path
import unittest
from test_gui_callbacks import CallbackContracts, Widget, Var
from test_vector_engine import config
from engine.evolution import run
from engine.vector_boolean import VectorBooleanNet
from engine import smith


class VectorGUI(CallbackContracts):
    def test_vector_recordings_and_eva_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            run(tmp,config(generations=1))
            def hook(app,ui):
                ui['refresh_all']();self.assertIn('revisione 1',ui['eva_status'].get())
                for source in ('Training','Test finale'):
                    ui['mode'].set('replay');ui['replay_source'].set(source);ui['next_step']()
                    self.assertEqual(ui['inspector'].brain['engine'],'boolean-vector/5')
                    self.assertIn('vector_state',ui['inspector'].frame)
                ui['refresh_diagnostics']()
                self.assertIn('associative_off',ui['diagnostics_text'].text)
                self.assertIn('SMITH',ui['diagnostics_text'].text)
                ui['close_window']()
            self.harness(Path(tmp),hook)

    def test_inspector_displays_actual_vectors_without_topology_rebuild(self):
        from gui.brain_inspector import BrainInspector
        view=BrainInspector.__new__(BrainInspector)
        view.info=Var();view.canvas=Widget();view.details=Widget()
        view.signature=None;view.brain=None;view.frame={}
        view.node_items=[];view.sensor_items=[];view.selected=None;view.rebuild_count=0
        n=VectorBooleanNet(3,neurons=5,shape=(4,4))
        n.decide([1]*15);n.feedback(-1)
        frame={'t':1,'action':0,'observation':[1]*15,'neural_state':n.state,**n.frame_data()}
        view.show(n.export(),frame,'Live Training');view.inspect(0)
        self.assertIn('Modulo 0',view.details.text)
        self.assertIn(format(n.vectors[0],'016b'),view.details.text)
        view.show(n.export(),dict(frame,t=2),'Live Training')
        self.assertEqual(view.rebuild_count,1)


class SupervisorTests(unittest.TestCase):
    def test_strategy_changes_with_observed_outcomes(self):
        state={};first=smith.choose(state);smith.observe(state,first,False)
        second=smith.choose(state);self.assertNotEqual(first,second)
        smith.observe(state,second,True)
        for name in ("represent","compact"):
            self.assertEqual(smith.choose(state),name)
            smith.observe(state,name,False)
        self.assertEqual(smith.choose(state),second)
        self.assertEqual(state['arms'][second],[1,1])

if __name__=='__main__':unittest.main()
