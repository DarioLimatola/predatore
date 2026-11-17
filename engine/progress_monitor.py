"""Read-only, paired training/validation monitoring. Never queries holdout.

Additional evaluations use fresh clones. Their results never enter ranking,
mutation, archive or compression. Monitoring adds measured work, not knowledge.
"""
from pathlib import Path
import time
from .boolean_net import BooleanNet
from .vault import read_json, atomic_json, append_jsonl


def completion_metrics(rows):
    success=[r['steps'] for r in rows if r['capture']]
    return {'capture_count':len(success),'episode_count':len(rows),
            'capture_steps_sum':sum(success),
            'successful_mean_steps':sum(success)/len(success) if success else None,
            'best_capture_steps':min(success) if success else None}


def population_completion(scored):
    count=sum(m.get('capture_count',0) for m,_ in scored)
    total=sum(m.get('capture_steps_sum',0) for m,_ in scored)
    best=[m['best_capture_steps'] for m,_ in scored if m.get('best_capture_steps') is not None]
    return {'population_successful_mean_steps':total/count if count else None,
            'population_best_capture_steps':min(best) if best else None}


class ProgressMonitor:
    def __init__(self,folder,config,every=10):
        self.folder=Path(folder);self.config=config;self.every=max(1,int(every))
        self.initialized=False;self.baselines={};self.history=[]
        self.seconds=0.;self.error=None

    def _initialize(self,generation):
        from .challenges import load_suites
        c=self.config
        from .evaluation import baseline_suite
        self.suites={split:load_suites(self.folder,split) for split in ('train','validation')}
        self.counts={'train':max(1,(c.episodes+3)//4),
                     'validation':max(1,(c.validation_episodes+3)//4)}
        self.baselines={split:{name:baseline_suite(name,suites,c.seed,split,self.counts[split],c.steps)
                              for name in ('random','greedy')} for split,suites in self.suites.items()}
        path=self.folder/'monitor/history.json'
        try:self.history=[r for r in read_json(path) if r['generation']<generation]
        except (OSError,ValueError,KeyError,TypeError):self.history=[]
        self.initialized=True

    def observe(self,summary,champion):
        from .evaluation import evaluate_suite
        from .telemetry import write_mailbox
        start=time.perf_counter()
        try:
            if not self.initialized:self._initialize(summary['generation'])
            generation=summary['generation']
            if getattr(self,'training_suites',None):
                from .evaluation import baseline_suite
                c=self.config
                self.baselines['train']={k:baseline_suite(k,self.training_suites,c.seed,'train',self.counts['train'],c.steps)
                                         for k in ('random','greedy')}
            snapshot={'generation':generation,'train':summary,'baselines':self.baselines,
                      'monitor_every':self.every,'monitor_seconds':self.seconds,
                      'note':'Addestramento/validazione; holdout riservato alla fine. Copie adattive isolate; nessun feedback delle prove al training.'}
            if generation==1 or generation%self.every==0 or generation==self.config.generations:
                clone=BooleanNet.restore(champion.export())
                clone.reset()
                clone.collect_experience=False
                validation=evaluate_suite(clone,self.suites['validation'],self.config.seed,'validation',
                                          self.counts['validation'],self.config.steps)
                self.history.append({'generation':generation,'validation':validation})
                atomic_json(self.folder/'monitor/history.json',self.history)
            if self.history:
                snapshot['validation']=self.history[-1]['validation']
                snapshot['validation_generation']=self.history[-1]['generation']
            self.seconds+=time.perf_counter()-start
            snapshot['monitor_seconds']=self.seconds
            write_mailbox(self.folder/'monitor/latest.json',snapshot)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            self.error=str(exc)
            print('Monitor non disponibile:',self.error,flush=True)
