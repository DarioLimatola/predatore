"""Persistent contextual training supervisor, separate from all world actors.

Two reproduction strategies are compared through observed balanced validation gains.
Forced exploration avoids permanently abandoning an untried strategy. Counts are
metadata, not motor decisions. No final-test result is consumed here.
"""
from .settings import Limits

def _choose(arms):
    rounds=sum(v[0] for v in arms.values())
    if rounds%Limits.SMITH_EXPLORATION_PERIOD==0:
        return ('consolidate','explore')[(rounds//Limits.SMITH_EXPLORATION_PERIOD)%2]
    for name in ('consolidate','explore'):
        if arms[name][0]==0:return name
    a,b=arms['consolidate'],arms['explore']
    return 'explore' if (b[1]+1)*(a[0]+2)>(a[1]+1)*(b[0]+2) else 'consolidate'


def choose(state,context=None):
    state.setdefault('arms',{'consolidate':[0,0],'explore':[0,0]})
    if context is None:return _choose(state['arms'])
    contexts=state.setdefault('contexts',{})
    arms=contexts.setdefault(str(context),{'consolidate':[0,0],'explore':[0,0]})
    return _choose(arms)


def observe(state,strategy,improved,context=None):
    choose(state)
    state['arms'][strategy][0]+=1
    state['arms'][strategy][1]+=int(improved)
    if context is not None:
        choose(state,context)
        state['contexts'][str(context)][strategy][0]+=1
        state['contexts'][str(context)][strategy][1]+=int(improved)
    state['last_strategy']=strategy
    state['last_improved']=bool(improved)


def context_for(summary):
    """Four observable training facts; no held-out result enters strategy choice."""
    low_capture=summary['capture_count']*Limits.SMITH_LOW_CAPTURE_DENOMINATOR<summary['episode_count']
    stall_count=summary.get('stall_count',round(summary['stall_rate']*summary['episode_count']))
    high_stall=stall_count*Limits.SMITH_HIGH_STALL_DENOMINATOR>=summary['episode_count']
    rejected=not summary['eva_promoted']
    conflict=summary.get('distillation',{}).get('conflicting_contexts',0)>0
    return int(low_capture)|(int(high_stall)<<1)|(int(rejected)<<2)|(int(conflict)<<3)


def record_run(state,run,history,starting_revision):
    """Update a bounded per-run memory using training/validation only.

    This upsert is safe at every checkpoint. It never feeds final tests back into
    learning, and a resumed run replaces its own partial summary deterministically.
    """
    record={'run':run,'starting_revision':starting_revision,'generations_seen':len(history),
            'promotions':sum(int(h['eva_promoted']) for h in history),
            'strict_improvements':sum(int(h['strict_improvement']) for h in history),
            'contexts_proposed':sum(h.get('distillation',{}).get('contexts_taught',0) for h in history),
            'contexts_consolidated':sum(h.get('distillation',{}).get('contexts_consolidated',0) for h in history),
            'distillation_promotions':sum(h['promotion_reason'] in ('contextual_distillation','teacher_distilled') and h['eva_promoted'] for h in history),
            'eva_learning_generations':sum(int(h.get('eva_lifetime_changed',False)) for h in history),
            'last_context':state.get('last_context',0),
            'last_train_capture_count':history[-1]['capture_count'],
            'last_train_episode_count':history[-1]['episode_count']}
    records=[r for r in state.setdefault('run_memory',[]) if r['run']!=run]
    state['run_memory']=(records+[record])[-Limits.SMITH_RUN_MEMORY:]


def describe(state):
    return {'role':Limits.SMITH_LABEL,
            'remembered_runs':len(state.get('run_memory',[])),
            'learned_contexts':len(state.get('contexts',{})),
            'strategy':state.get('last_strategy'),
            'scope':'Learns which of two combined reproduction/distillation strategies yields balanced validation gains in each training context.'}
