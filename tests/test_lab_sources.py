"""Native Tk tests for the P6 interaction and data contracts."""
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

from engine.evolution import run
from engine.telemetry import LiveBuffer, LiveObserver, write_mailbox
from gui.arena_state import frame_summary
from gui.laboratory import launch
from test_vector_engine import config


class SourceContracts(unittest.TestCase):
    def test_initial_frame_does_not_infer_visibility_or_capture(self):
        label=frame_summary({'t':0,'predator':[0,0],'prey':[1,1]})
        self.assertNotIn('visibile',label)
        self.assertNotIn('catturata',label)

    def test_locked_mailbox_recovers_and_atomic_replacement_advances(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'live/latest.json'
            packet={'run_id':'run','episode_id':1,'sequence':1,'timestamp':1,'frames':[{'t':1}]}
            write_mailbox(path,packet)
            buffer=LiveBuffer()
            with patch.object(Path,'read_text',side_effect=PermissionError('locked')):
                self.assertFalse(buffer.poll(tmp))
                self.assertEqual(buffer.error,'PermissionError')
            self.assertTrue(buffer.poll(tmp))
            with patch.object(Path,'read_text',side_effect=AssertionError('duplicate read')):
                self.assertFalse(buffer.poll(tmp))
            write_mailbox(path,{**packet,'sequence':2,'frames':[{'t':2}]})
            self.assertTrue(buffer.poll(tmp))
            self.assertEqual(buffer.latest()[1]['t'],2)
            path.unlink()
            self.assertFalse(buffer.poll(tmp))
            self.assertEqual(buffer.current[1]['t'],2)

    def test_native_sources_extended_and_credit(self):
        try:
            probe=tk.Tk()
            probe.destroy()
        except tk.TclError as exc:
            self.skipTest(f'Display unavailable: {exc}')
        with tempfile.TemporaryDirectory() as tmp:
            c=config(generations=1,seed=722,scenario_set='extended',relation_credit_horizon=3)
            exp=Path(tmp)/'EvolutionVault/experiments/EXP_seed722_size6'
            observer=LiveObserver(exp,interval=.02)
            try:run(tmp,c,observer=observer)
            finally:observer.close()

            def check(app,ui):
                errors=[]
                app.report_callback_exception=lambda *args:errors.append(args)
                ui['tabs'].select(ui['panels']['arena'])
                app.update()
                self.assertEqual(len(ui['tabs'].tabs()),5)
                ui['poll_live']()
                self.assertIsNotNone(ui['live_buffer'][0].current)
                self.assertIn('training reale',ui['arena_status'].get())
                self.assertNotIn('preview_actors',ui)
                ui['mode'].set('replay')
                ui['replay_all'].set(True);ui['load_replay']();app.update()
                self.assertGreaterEqual(len(ui['replay_group']),2)
                ui['replay_all'].set(False)
                ui['replay_source'].set('Test finale')
                label=next(v for v in ui['replay_combo']['values'] if 'Ricompensa ritardata' in v)
                ui['replay_name'].set(label)
                ui['seek_replay'](3)
                self.assertFalse(ui['playing'][0])
                frame=ui['replay_frames'][0][ui['replay_index'][0]]
                self.assertEqual(frame,ui['inspector'].frame)
                self.assertIn('TEST finale',ui['arena_status'].get())
                self.assertIn('Ricompensa ritardata',ui['arena_status'].get())
                self.assertIn('relation_credit',frame)
                self.assertIn('world_events',frame)
                ui['tabs'].select(ui['panels']['brain'])
                inspector=ui['inspector']
                inspector.group.set('Uscite')
                inspector.zoom.set('125%')
                inspector.redraw()
                self.assertEqual(set(inspector.node_items),set(inspector.brain['outputs']))
                inspector.inspect(inspector.brain['outputs'][0])
                self.assertIn('Credito relazionale',inspector.details.get('1.0','end'))
                self.assertIn('Maschera scritture LUT',inspector.details.get('1.0','end'))
                self.assertFalse(errors)
                ui['close_window']()
            launch(tmp,test_hook=check)


if __name__=='__main__':
    unittest.main()
