"""Deduplicated teacher consensus, written independently of personal evidence."""
import hashlib
import json
from .limits import CoreLimits as L
from .memory import ContextMemory, greater, increment


def distill_into(student, events, focus_terminal=False):
    codec=student.context_codec
    rows={}; terminal={}; identities={}; seen=set(); sources=set()
    mask=codec.action_mask
    for event in events:
        identifier=event.get('event_id') or hashlib.sha256(json.dumps(event,sort_keys=True).encode()).hexdigest()
        if identifier in seen:continue
        seen.add(identifier)
        source=event.get('source','external'); sources.add(source)
        def record(key,action,positive,goal=False):
            if not codec.validate(key) or not 0<=action<codec.actions:return
            row=rows.setdefault((source,key),ContextMemory.empty())
            identities.setdefault(key,set()).add(identifier)
            if goal:terminal[source,key]=terminal.get((source,key),0)|(1<<action)
            for _ in range(L.CONTEXT_TERMINAL_CREDIT if goal else 1):
                increment(row[0 if positive else 1],1<<action,mask)
        if event['feedback'] in (1,2):
            key=codec.encode(event['observation'],event['history']) if 'history' in event else event.get('context_key')
            record(key,event['action'],event['feedback']==1)
        if event.get('terminal_success'):
            for payload,action in event.get('eligible_contexts',[]):
                key=codec.encode(payload['observation'],payload['history']) if isinstance(payload,dict) else payload
                record(key,action,True,True)
    votes={}; owners={}
    for (source,key),row in rows.items():
        goal=terminal.get((source,key),0)
        yes,no=greater(*row,mask),greater(row[1],row[0],mask)
        if focus_terminal:yes&=goal
        total=votes.setdefault(key,ContextMemory.empty())
        increment(total[0],yes,mask);increment(total[1],no,mask)
        for _ in range(L.CONTEXT_TERMINAL_CREDIT-1):increment(total[0],yes&goal,mask)
        owners.setdefault(key,set()).add(source)
    memory=student.contextual_memory
    taught=conflicts=0; taught_sources=set()
    for key in sorted(votes,key=lambda k:(-memory.supported(votes[k],mask).bit_count(),k)):
        row=votes[key]
        good=memory.supported(row,mask);bad=memory.supported(row[::-1],mask)
        if not good|bad and key not in memory.teachers:continue
        supported=good|bad
        row=[[plane & supported for plane in group] for group in row]
        signature=hashlib.sha256(json.dumps([row,sorted(identities[key]),focus_terminal]).encode()).hexdigest()
        old=memory.evidence(key)
        if memory.set_teacher(key,row,owners[key],signature):
            if old and old[0]&bad:conflicts+=1
            taught+=1;taught_sources.update(owners[key]);student.sync_associative(key)
        if taught>=min(memory.capacity,L.DISTILL_CONTEXT_BUDGET):break
    student.reset()
    return {'events_considered':len(seen),'contributors_considered':len(sources),
            'contexts_observed':len(votes),'contexts_taught':taught,'contributors_taught':len(taught_sources),
            'conflicting_contexts':conflicts,'focus_terminal':focus_terminal}
