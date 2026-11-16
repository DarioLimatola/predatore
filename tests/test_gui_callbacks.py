"""Callback contract tests with widget doubles. Not native rendering tests.

A real Tk smoke test is also provided and runs automatically when DISPLAY exists
(or on Windows). The doubles exercise the complete launch/callback wiring here.
"""
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import tkinter as tk
from tkinter import ttk

from engine.evolution import run
from engine.telemetry import LiveObserver
from test_vector_engine import config
from gui.arena_state import preview_settings, episode_finished


class Var:
    def __init__(self,*args,value=None,**kwargs):self.value=value;self.callbacks=[]
    def get(self):return self.value
    def set(self,value):
        self.value=value
        for callback in self.callbacks:callback()
    def trace_add(self,kind,callback):self.callbacks.append(callback)


class Widget:
    serial=0
    def __init__(self,*args,**kwargs):
        self.options=dict(kwargs);self.children=[];self.items={};self.text='';self.selection=''
        self.callbacks={};self.tk=self
        if args and isinstance(args[0],Widget):args[0].children.append(self)
    def __str__(self):return str(id(self))
    def __setitem__(self,key,value):self.options[key]=value
    def __getitem__(self,key):return self.options[key]
    def pack(self,*a,**k):pass
    grid=pack;pack_propagate=pack;columnconfigure=pack;title=pack;geometry=pack;minsize=pack
    theme_use=pack;map=pack;heading=pack;column=pack;bind=pack;tag_bind=pack;see=pack;protocol=pack
    yview=pack;xview=pack;set=pack;itemconfigure=pack;coords=pack
    def configure(self,**kwargs):self.options.update(kwargs)
    def add(self,child,**kwargs):
        if not self.selection:self.selection=str(child)
    def select(self,child=None):
        if child is not None:self.selection=str(child)
        return self.selection
    def winfo_children(self):return list(self.children)
    def winfo_width(self):return 1100
    def winfo_height(self):return 500
    def create_oval(self,*a,**k):
        Widget.serial+=1;self.items[str(Widget.serial)]=k;return str(Widget.serial)
    create_rectangle=create_oval;create_text=create_oval;create_line=create_oval
    def insert(self,*args,**kwargs):
        if 'values' in kwargs:return self.create_oval(values=kwargs['values'])
        self.text+=str(args[-1])
    def delete(self,*args):
        if args and args[0] in self.items:self.items.pop(args[0])
        else:self.text='';self.items={}
    def get_children(self):return list(self.items)
    def after(self,delay,callback):
        key=str(len(self.callbacks)+1);self.callbacks[key]=callback;return key
    def after_cancel(self,key):self.callbacks.pop(key,None)
    def call(self,*args):return tuple(self.callbacks)
    def destroy(self):self.children=[]
    def mainloop(self):pass


class Style(Widget):
    def configure(self,*args,**kwargs):pass


class CallbackContracts(unittest.TestCase):
    def harness(self,root,hook):
        from gui.laboratory import launch
        with ExitStack() as stack:
            for name in ('Tk','Frame','Label','Canvas','Text'):
                stack.enter_context(patch.object(tk,name,Widget))
            for name in ('StringVar','IntVar','BooleanVar'):
                stack.enter_context(patch.object(tk,name,Var))
            for name in ('Frame','Label','LabelFrame','Button','Checkbutton','Radiobutton','Combobox',
                         'Entry','Spinbox','Scale','Notebook','Treeview','Scrollbar'):
                stack.enter_context(patch.object(ttk,name,Widget))
            stack.enter_context(patch.object(ttk,'Style',Style))
            # The inspector base ttk.Frame was bound at import: override just its initializer.
            from gui.brain_inspector import BrainInspector
            original=BrainInspector.__init__
            stack.enter_context(patch.object(BrainInspector,'__init__',lambda self,*a,**k:None))
            inspector=Widget()
            inspector.frame={};inspector.brain=None;inspector.rebuild_count=0;inspector.signature=None
            def show(brain,frame=None,source=''):
                inspector.frame=frame or {};inspector.brain=brain
                signature=str(brain.get('nodes')) if brain else None
                if signature!=inspector.signature:inspector.rebuild_count+=1;inspector.signature=signature
            inspector.show=show
            stack.enter_context(patch('gui.laboratory.BrainInspector',return_value=inspector))
            launch(root,test_hook=hook)

    def test_empty_start_close_and_no_post_close_callbacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            def hook(app,ui):
                self.assertEqual(ui['mode'].get(),'live')
                self.assertIn('non disponibile',ui['arena_status'].get())
                ui['close_window']();self.assertTrue(ui['closing'][0])
                ui['poll_live']();ui['periodic']();ui['schedule_tick']()
                self.assertFalse(app.callbacks)
            self.harness(Path(tmp),hook)

    def test_sources_pause_step_and_deleted_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=Path(tmp)/'EvolutionVault/experiments/EXP_seed29_size6'
            observer=LiveObserver(exp,interval=.02)
            run(tmp,config(seed=29,generations=1),observer=observer);observer.close()
            def hook(app,ui):
                ui['next_step']()
                current=ui['live_buffer'][0].current
                self.assertIsNotNone(current)
                self.assertEqual(ui['inspector'].frame,current[1])
                ui['toggle_play']();frame=ui['inspector'].frame
                ui['poll_live']();self.assertEqual(ui['inspector'].frame,frame)
                ui['step_observed']();self.assertFalse(ui['playing'][0])
                ui['mode'].set('preview');ui['next_step']()
                self.assertIn('Simulazioni indipendenti',ui['arena_status'].get())
                self.assertTrue(ui['inspector'].brain)
                ui['mode'].set('replay');ui['next_step']()
                self.assertIn('Recorded Replay',ui['arena_status'].get())
                self.assertEqual(ui['inspector'].frame,ui['replay_frames'][0][ui['replay_index'][0]])
                ui['refresh_diagnostics']()
                self.assertIn('plasticity_off',ui['diagnostics_text'].text)
                shutil.rmtree(exp)
                ui['refresh_all']();ui['mode'].set('live');ui['poll_live']();ui['draw_arena']()
                self.assertIn('non disponibile',ui['arena_status'].get())
                ui['close_window']()
            self.harness(Path(tmp),hook)


class ArenaFaultTolerance(unittest.TestCase):
    def test_orphan_directory_and_removed_config_are_handled(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'orphan'
            folder.mkdir()
            (folder/'gui_worker.log').write_text('hello')
            self.assertIsNone(preview_settings(folder))
            self.assertFalse(episode_finished([],100))

            (folder/'environment').mkdir()
            (folder/'config.json').write_text(json.dumps({'steps':21,'seed':7}))
            (folder/'environment'/'holdout.json').write_text(
                json.dumps({'size':6,'obstacles':[]}))
            settings=preview_settings(folder)
            self.assertEqual(settings['limit'],21)
            (folder/'config.json').unlink()
            self.assertIsNone(preview_settings(folder))

            class MockWorld:
                steps=21
                captured=False

            self.assertTrue(episode_finished([{'world':MockWorld()}],settings['limit']))


class NativeTkSmoke(unittest.TestCase):
    def test_native_start_modes_and_close_when_display_available(self):
        try:
            probe=tk.Tk();probe.destroy()
        except tk.TclError as exc:
            self.skipTest('Native Tk display unavailable: '+str(exc))
        from gui.laboratory import launch
        with tempfile.TemporaryDirectory() as tmp:
            run(tmp,config(generations=1))
            def hook(app,ui):
                errors=[]
                app.report_callback_exception=lambda *args:errors.append(args)
                app.update()
                for mode in ('preview','replay','live'):
                    ui['mode'].set(mode);ui['step_observed']();app.update()
                ui['close_window']()
                self.assertEqual(errors,[])
            launch(tmp,test_hook=hook)

if __name__=='__main__':unittest.main()
