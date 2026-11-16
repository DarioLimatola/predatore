"""Shared observable contexts, Boolean evidence and multi-contributor distillation.

No world coordinates, map/task identity, neural private state or future data enter
an action's context. Terminal outcomes are used only after they actually occur.
"""
from collections import OrderedDict
from .settings import Limits


class ContextCodec:
    def __init__(self,sensors=Limits.SENSOR_BITS,history=Limits.CONTEXT_HISTORY):
        self.sensors=sensors;self.history=history
        self.value_bits=sensors+history*(sensors+Limits.ACTION_BITS)
        self.bits=self.value_bits+history

    def encode(self,current,history):
        if not 0<=current<(1<<self.sensors):raise ValueError('Observation outside context schema')
        past=list(history)[-self.history:]
        key=current;shift=self.sensors
        for observation,action in reversed(past):
            key |= observation << shift;shift+=self.sensors
            key |= action << shift;shift+=Limits.ACTION_BITS
        key |= ((1<<len(past))-1) << self.value_bits
        return str(key)

    def validate(self,key):
        return isinstance(key,str) and key.isdecimal() and 0<=int(key)<(1<<self.bits)


def increment(planes,lanes):
    """One saturating increment, eight Boolean counters in parallel."""
    saturated=Limits.ACTION_MASK
    for plane in planes:saturated &= plane
    carry=lanes & ~saturated & Limits.ACTION_MASK
    for i,old in enumerate(planes):
        planes[i]=old ^ carry
        carry &= old


def greater(a,b):
    equal=Limits.ACTION_MASK;result=0
    for x,y in zip(reversed(a),reversed(b)):
        result |= equal & x & ~y
        equal &= ~(x ^ y) & Limits.ACTION_MASK
    return result & Limits.ACTION_MASK


def difference(a,b):
    borrow=0;result=[]
    for x,y in zip(a,b):
        result.append((x ^ y ^ borrow) & Limits.ACTION_MASK)
        borrow=((~x & (y | borrow)) | (y & borrow)) & Limits.ACTION_MASK
    return result


def counter_value(planes,action):
    return sum(((plane>>action)&1)<<i for i,plane in enumerate(planes))


class ContextMemory:
    """Bounded RAM; each row has positive/negative bit-plane counters per action."""
    def __init__(self,capacity):
        self.capacity=capacity;self.rows=OrderedDict()

    @staticmethod
    def empty():return [[0]*Limits.CONTEXT_COUNTER_BITS,[0]*Limits.CONTEXT_COUNTER_BITS]

    @staticmethod
    def supported(row):
        positive,negative=row
        minimum=[Limits.ACTION_MASK if (Limits.CONTEXT_MIN_SUPPORT>>i)&1 else 0
                 for i in range(Limits.CONTEXT_COUNTER_BITS)]
        enough=~greater(minimum,positive) & Limits.ACTION_MASK
        return enough & greater(positive,negative)

    def update(self,key,action,positive,weight=1):
        row=self.rows.setdefault(str(key),self.empty())
        for _ in range(weight):increment(row[0 if positive else 1],1<<action)
        self.rows.move_to_end(str(key))
        if len(self.rows)>self.capacity:self.rows.popitem(last=False)

    def choose(self,key,proposed):
        row=self.rows.get(str(key))
        if row is None:return None
        winners=self.supported(row)
        if not winners:
            rejected=self.supported([row[1],row[0]])
            if not rejected & (1<<proposed):return None
            winners=~rejected & Limits.ACTION_MASK
            if not winners:return None
            for offset in range(Limits.ACTIONS):
                action=(proposed+offset)&(Limits.ACTIONS-1)
                if winners & (1<<action):return action
        # Max positive-minus-negative margin through parallel Boolean bit planes.
        for plane in reversed(difference(*row)):
            preferred=winners & plane
            if preferred:winners=preferred
        for offset in range(Limits.ACTIONS):
            action=(proposed+offset)&(Limits.ACTIONS-1)
            if winners & (1<<action):return action
        return None

    def set_row(self,key,row):
        self.rows[str(key)]=[list(row[0]),list(row[1])]
        self.rows.move_to_end(str(key))
        if len(self.rows)>self.capacity:self.rows.popitem(last=False)

    def export(self):return [[k,[list(row[0]),list(row[1])]] for k,row in self.rows.items()]

    def restore(self,data,codec):
        if len(data)>self.capacity:raise ValueError('Context memory capacity exceeded')
        for key,row in data:
            if not codec.validate(key) or len(row)!=2 or any(len(p)!=Limits.CONTEXT_COUNTER_BITS for p in row):
                raise ValueError('Invalid contextual memory schema')
            if any(type(x) is not int or not 0<=x<=Limits.ACTION_MASK for p in row for x in p):
                raise ValueError('Invalid Boolean evidence plane')
            self.set_row(key,row)


def distill_into(student,events,focus_terminal=False):
    """Aggregate each contributor's contextual opinion once, then teach the student.

    Reprocessing the same bounded archive does not add votes repeatedly. Terminal
    traces carry extra evidence; conflicting opinions remain visible and bounded.
    Evidence is a confidence heuristic, not independent statistical significance.
    """
    contributors={};terminals={};sources=set();considered=0
    for event in events:
        source=event['source'];sources.add(source);considered+=1
        key=event.get('context_key');action=event['action'];feedback=event['feedback']
        if student.context_codec.validate(key) and feedback in (1,2):
            rows=contributors.setdefault(source,{})
            row=rows.setdefault(key,ContextMemory.empty())
            increment(row[0 if feedback==1 else 1],1<<action)
        if event.get('terminal_success'):
            rows=contributors.setdefault(source,{})
            for key,act in event.get('eligible_contexts',[]):
                if not student.context_codec.validate(key) or not 0<=act<Limits.ACTIONS:continue
                row=rows.setdefault(key,ContextMemory.empty())
                for _ in range(Limits.CONTEXT_TERMINAL_CREDIT):increment(row[0],1<<act)
                terminal=terminals.setdefault(source,{})
                terminal[key]=terminal.get(key,0)|(1<<act)
    votes={};owners={}
    for source,rows in contributors.items():
        for key,row in rows.items():
            terminal=terminals.get(source,{}).get(key,0)
            yes=greater(row[0],row[1]);no=greater(row[1],row[0])
            if focus_terminal:yes &= terminal
            aggregate=votes.setdefault(key,ContextMemory.empty())
            increment(aggregate[0],yes);increment(aggregate[1],no)
            for _ in range(Limits.CONTEXT_TERMINAL_CREDIT-1):increment(aggregate[0],yes & terminal)
            owners.setdefault(key,set()).add(source)
    taught=0;conflicts=0;taught_sources=set()
    # Deterministic ordering and a hard write budget; unchanged rows cost no writes.
    ranked=sorted(votes,key=lambda key:(-ContextMemory.supported(votes[key]).bit_count(),key))
    for key in ranked:
        row=votes[key]
        if not (ContextMemory.supported(row) | ContextMemory.supported([row[1],row[0]])):continue
        previous=student.contextual_memory.rows.get(key)
        if previous==row:continue
        if previous and ContextMemory.supported(previous) & ~ContextMemory.supported(row):conflicts+=1
        student.contextual_memory.set_row(key,row)
        taught_sources.update(owners[key]);taught+=1
        if taught>=min(Limits.DISTILL_CONTEXT_BUDGET,student.contextual_memory.capacity):break
    student.reset()
    return {'events_considered':considered,'contributors_considered':len(sources),
            'contexts_observed':len(votes),'contexts_taught':taught,
            'contributors_taught':len(taught_sources),'conflicting_contexts':conflicts,
            'focus_terminal':focus_terminal}
