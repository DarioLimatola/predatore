"""Boolean vector recurrent network + learned RAM action decoder.

All decision state, truth tables, write masks, addresses and evidence are bits.
Numeric rewards are converted at the environment boundary. No coordinates or
hand-written movement policy enter this controller. Integers pack bit arrays.
"""
from collections import OrderedDict, deque
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import math
import random

from .graph import BooleanNet, Neuron, gate

from .limits import CoreLimits as Limits
from .context import ContextCodec
from .memory import ContextMemory, add
from .hdc import BinarySpace, AssociativeMemory
from .tensor import validate_shape, transform, mutate_shape, OPERATORS

ENGINE = Limits.ENGINE


@lru_cache(maxsize=32)
def _width_mask(width):
    return (1 << width)-1


def pack(bits):
    value = 0
    for i, bit in enumerate(bits):
        if bit not in (0, 1, False, True):
            raise ValueError('Expected Boolean values')
        value |= int(bit) << i
    return value


def unpack(value, width):
    return [(value >> i) & 1 for i in range(width)]


def masked_write(old, values, write_mask, width):
    return ((old & ~write_mask) | (values & write_mask)) & _width_mask(width)


def rotate(value, shift, width):
    shift %= width
    mask = _width_mask(width)
    if not shift:
        return value & mask
    return ((value << shift) | (value >> (width-shift))) & mask


class VectorBooleanNet(BooleanNet):
    """Sparse graph of n-dimensional bit modules, synchronous bounded ticks.

    RAM evidence is part of the neural decoder, not an external movement rule.
    Persistent evidence and circuits survive reset(); working state does not.
    """
    def __init__(self, seed=1, sensors=Limits.SENSOR_BITS, neurons=Limits.DEFAULT_BASE_NEURONS, shape=Limits.DEFAULT_SHAPE,
                 memory_capacity=Limits.DEFAULT_MEMORY, internal_ticks=Limits.DEFAULT_TICKS, trace_depth=Limits.DEFAULT_TRACE,
                 action_bits=Limits.ACTION_BITS, context_history=Limits.CONTEXT_HISTORY,
                 hdc_width=Limits.HDC_WIDTH, hdc_capacity=Limits.HDC_CAPACITY):
        shape=validate_shape(shape)
        width=math.prod(shape)
        if not (0<=neurons<=Limits.MAX_NEURONS and neurons*width<=Limits.MAX_STATE_BITS
                and 1<=internal_ticks<=Limits.MAX_TICKS and 1<=trace_depth<=Limits.MAX_TRACE):
            raise ValueError('Invalid resource budget')
        self.context_codec=ContextCodec(sensors,context_history,action_bits)
        super().__init__(seed,sensors,neurons,action_bits)
        self.shape = tuple(shape); self.width = width; self.mask = (1 << width)-1
        self.memory_capacity = memory_capacity; self.internal_ticks = internal_ticks
        self.trace_depth = trace_depth
        rng = random.Random(seed ^ 0xB17)
        self.rules = [self._rule(n, rng) for n in self.nodes]
        self.output_lanes = [rng.randrange(width) for _ in range(self.action_bits)]
        self.sensor_encoding = "mixed"
        self.short_memory = ContextMemory(memory_capacity,self.actions)
        self.knowledge = self.short_memory.rows
        self.predictions = OrderedDict()
        self.contextual_memory = ContextMemory(memory_capacity,self.actions)
        self.associative_memory = AssociativeMemory(self.context_codec,min(hdc_capacity,memory_capacity),hdc_width)
        self.hdc_enabled=True;self.sparse_enabled=True;self.tensor_enabled=True
        self.contextual_enabled = True
        self.plasticity_enabled = True; self.memory_enabled = True
        self.associative_enabled = True
        self.circuit_plasticity_enabled = True
        self.collect_experience = False
        self.experience_sink = None
        self.circuit_evidence = {}
        self._ancestor_cache = None
        self.reset()

    def _rule(self, node, rng):
        return {'shift_a':rng.randrange(self.width),'shift_b':rng.randrange(self.width),
                'hold':rng.randrange(self.sensors+len(self.nodes)) if rng.randrange(4)==0 else -1,
                'write':self.mask,'shape':list(self.shape),
                'operator_a':'rotate','operator_b':'rotate','axis_a':0,'axis_b':0,'delay_a':0,'delay_b':0,
                'lut':[self.mask if gate(node.op,a,b) else 0 for a,b in ((0,0),(0,1),(1,0),(1,1))]}

    def reset(self):
        self.vectors = [0]*len(self.nodes)
        self.state = [0]*len(self.nodes) # Lane-zero projection for displays.
        self.context = 0; self.context_valid = 0
        self.observable_history = deque(maxlen=Limits.MAX_CONTEXT_HISTORY)
        self.context_trace = deque(maxlen=self.trace_depth)
        self.current_context_key = None
        self.trace = deque(maxlen=self.trace_depth)
        self.circuit_trace = deque(maxlen=self.trace_depth)
        self.last = None; self.last_updates = []; self.update_count = 0
        self.captured=False; self.next_observation=0
        self.last_prediction_error = None
        self.last_decoder = {}
        self.before_vectors = list(self.vectors)
        self.last_activity = []
        self.input_history=deque(maxlen=Limits.MAX_DELAY)
        self.last_payload=None
        self.computed_nodes=()
        self.decoder_counts=dict.fromkeys(('circuit','short','context','hdc'),0)

    def _address(self, observation):
        current = pack(observation)
        # Validity distinguishes initial context from an actual all-zero observation.
        return current | (self.context << self.sensors) | (self.context_valid << (2*self.sensors))

    def decide(self, observation):
        if len(observation) != self.sensors:
            raise ValueError('sensor width mismatch')
        obs = pack(observation)
        address = obs | (self.context << self.sensors) | (self.context_valid << (2*self.sensors))
        history=list(self.observable_history) if self.memory_enabled else []
        self.current_context_key = self.context_codec.encode(obs,history)
        self.last_payload={'observation':obs,'history':history}
        self.before_vectors = list(self.vectors)
        activity = []
        collect_activity = self.collect_experience and self.circuit_plasticity_enabled
        ancestors=self._output_ancestors()
        activity_nodes=ancestors if collect_activity else ()
        indices=ancestors if self.sparse_enabled else tuple(i for i,n in enumerate(self.nodes) if n.active)
        self.computed_nodes=indices
        mask = self.mask
        rules = self.rules
        nodes = self.nodes
        outputs = self.outputs
        output_lanes = self.output_lanes
        width = self.width
        for _ in range(self.internal_ticks):
            old = self.vectors if self.memory_enabled else [0]*len(self.nodes)
            # A sensor vector mixes its bit with a shifted whole observation.
            # The encoding is fixed, shared and contains only actual sensor data.
            if self.sensor_encoding == 'broadcast':
                inputs = [mask if bit else 0 for bit in observation]
            else:
                inputs = []
                for index, bit in enumerate(observation):
                    shift = index % width
                    mixed = (obs & mask) if not shift else (
                        (obs << shift) | (obs >> (width-shift))) & mask
                    inputs.append(mixed ^ (mask if bit else 0))
            inputs.extend(old)
            new = [0]*len(nodes)
            tick_activity = {} if collect_activity else None
            for i in indices:
                node,r=nodes[i],rules[i]
                def operand(side,source):
                    delay=r.get('delay_'+side,0) if self.memory_enabled else 0
                    value=inputs[source] if not delay else (self.input_history[-delay][source] if len(self.input_history)>=delay else 0)
                    op=r.get('operator_'+side,'rotate')
                    shift=r['shift_'+side]
                    if not self.tensor_enabled or op=='rotate':
                        return rotate(value,shift,width)
                    return transform(value,tuple(r.get('shape',self.shape)),op,r.get('axis_'+side,0),shift)
                a,b=operand('a',node.a),operand('b',node.b)
                na,nb = a ^ mask,b ^ mask
                t = r['lut']
                hold = inputs[r['hold']] if r['hold'] >= 0 and self.memory_enabled else 0
                write = r['write'] & ~hold & mask
                if collect_activity and i in activity_nodes:
                    tick_activity[i] = (na & nb & write, na & b & write,
                                        a & nb & write, a & b & write)
                value = ((na & nb & t[0]) | (na & b & t[1]) |
                         (a & nb & t[2]) | (a & b & t[3]))
                new[i]=((old[i] & ~write) | (value & write)) & mask
            if collect_activity:
                activity = tick_activity
            self.vectors = new
            if self.memory_enabled:self.input_history.append(inputs)
        self.last_activity = activity if collect_activity else []
        self.state = [v & 1 for v in self.vectors]
        proposed = sum(((self.vectors[n] >> lane) & 1) << bit
                       for bit,(n,lane) in enumerate(zip(outputs,output_lanes)))
        action=proposed;origin='circuit';contextual_action=None
        if self.associative_enabled:
            short=self.short_memory.choose(str(address),action)
            if short is not None:action=short;origin='short'
            if self.contextual_enabled:
                key=self.current_context_key
                contextual_action=self.contextual_memory.choose(key,action)
                if contextual_action is not None:action=contextual_action;origin='context'
                elif self.hdc_enabled:
                    anchor=obs & self.associative_memory.sensor_mask
                    for neighbor in list(self.associative_memory.buckets.get(anchor,{}))[-Limits.HDC_CANDIDATES:]:
                        self.sync_associative(neighbor)
                    e=self.contextual_memory.evidence(key)
                    similar=self.associative_memory.choose(key,action,e[1] if e else 0,self.contextual_memory.rows)
                    if similar is not None:action=similar;origin='hdc'
        self.decoder_counts[origin]+=1
        self.last_decoder={'context_key':self.current_context_key,'context_override':contextual_action,
                           'proposed':proposed,'address':str(address),'action':action,'source':origin}
        if collect_activity:
            ancestors = self._output_ancestors()
            active_patterns = []
            for node in sorted(ancestors):
                for pattern, lanes in enumerate(self.last_activity.get(node, ())):
                    lanes &= self.rules[node]['write']
                    if lanes:
                        active_patterns.append((node, pattern, lanes))
            self.circuit_trace.append((action == proposed, active_patterns))
        self.context_trace.append((self.last_payload,action))
        if self.memory_enabled:self.observable_history.append((obs,action))
        else:self.observable_history.clear()
        self.last = (address,action,obs,self.context,self.context_valid)
        self.trace.append(self.last)
        self.context = obs if self.memory_enabled else 0
        self.context_valid = int(self.memory_enabled)
        return action

    def sync_associative(self,key):
        e=self.contextual_memory.evidence(key)
        positive,negative=e[:2] if e else (0,0)
        old=self.associative_memory.entries.get(key)
        if old is None or old[2:]!=(positive,negative):
            self.associative_memory.observe(key,positive,negative)

    def learn(self,address,action,positive):
        key=str(address)
        self.short_memory.update(key,action,positive)
        self.update_count+=1
        self.last_updates.append({'address':key,'action':action,'positive':bool(positive)})

    def transition(self, next_observation, captured=False):
        self.next_observation = pack(next_observation)
        self.captured = bool(captured)

    def clear_circuit_evidence(self):
        self.circuit_evidence.clear()
        self.circuit_trace.clear()

    def _output_ancestors(self):
        if self._ancestor_cache is not None:
            return self._ancestor_cache
        ancestors=set()
        pending=list(self.outputs)
        while pending:
            index=pending.pop()
            node=self.nodes[index]
            if index in ancestors or not node.active:continue
            ancestors.add(index)
            for source in (node.a,node.b,self.rules[index]['hold']):
                if source>=self.sensors:pending.append(source-self.sensors)
        self._ancestor_cache=tuple(sorted(ancestors))
        return self._ancestor_cache

    def invalidate(self):
        self._ancestor_cache=None

    def _reinforce_circuit_bits(self, node, pattern, lanes, positive):
        low_pos, high_pos, low_neg, high_neg = self.circuit_evidence.get(
            (node, pattern), (0, 0, 0, 0))
        low, high = (low_pos, high_pos) if positive else (low_neg, high_neg)
        write = lanes & ~(low & high) & self.mask
        low, high = low ^ write, high ^ (low & write)
        if positive:
            values = (low, high, low_neg, high_neg)
        else:
            values = (low_pos, high_pos, low, high)
        self.circuit_evidence[(node, pattern)] = values

    def local_circuit_candidates(self, limit=2):
        ranked = []
        for (node, pattern), (low_pos, high_pos, low_neg, high_neg) in self.circuit_evidence.items():
            supported = low_pos & high_pos
            opposed = low_neg & high_neg & ~supported & self.mask
            if opposed:
                ranked.append((opposed.bit_count(), node, pattern, opposed & -opposed))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [(node, pattern, lane.bit_length()-1)
                for _, node, pattern, lane in ranked[:max(0, limit)]]

    def apply_local_circuit_candidate(self, candidate):
        if len(candidate) != 3:
            raise ValueError('Invalid local circuit candidate')
        node, pattern, lane = candidate
        if (type(node) is not int or not 0 <= node < len(self.nodes) or
                type(pattern) is not int or not 0 <= pattern < 4 or
                type(lane) is not int or not 0 <= lane < self.width):
            raise ValueError('Local circuit candidate outside network bounds')
        self.rules[node]['lut'][pattern] ^= 1 << lane

    def feedback(self, reward):
        self.last_updates = []
        if self.last is None:
            return
        address,action,obs,context,valid = self.last
        positive = reward > 0
        negative = reward < 0
        next_obs = getattr(self,'next_observation',obs)
        prediction_key = f'{address}:{action}'
        predicted = self.predictions.get(prediction_key)
        self.last_prediction_error = None if predicted is None else predicted ^ next_obs
        if self.plasticity_enabled and self.circuit_plasticity_enabled and reward != 0:
            credited = {}
            for aligned, active_patterns in reversed(self.circuit_trace):
                if not aligned:
                    continue
                for node, pattern, lanes in active_patterns:
                    key = (node, pattern)
                    credited[key] = credited.get(key, 0) | lanes
            for (node, pattern), lanes in credited.items():
                self._reinforce_circuit_bits(node, pattern, lanes, positive)
        if self.plasticity_enabled:
            if positive or negative:
                self.learn(address,action,positive)
                if self.current_context_key is not None:
                    self.contextual_memory.update(self.current_context_key,action,positive)
                    self.sync_associative(self.current_context_key)
            # Bounded terminal-success eligibility trace. It is experimental credit
            # assignment, not a claim that every preceding action caused the capture.
            if getattr(self,'captured',False):
                for payload,act in self.context_trace:
                    key=self.context_codec.encode(payload['observation'],payload['history'])
                    self.contextual_memory.update(key,act,True,Limits.CONTEXT_TERMINAL_CREDIT,channel='terminal')
                    self.sync_associative(key)
            self.predictions[prediction_key] = next_obs
            self.predictions.move_to_end(prediction_key)
            if len(self.predictions) > self.memory_capacity:
                self.predictions.popitem(last=False)
        if self.collect_experience and self.experience_sink is not None:
            self.experience_sink({'context_key':self.current_context_key,
                                  'eligible_contexts':list(self.context_trace) if self.captured else [],
                                  'observation':obs,'history':self.last_payload['history'],'previous':context,'valid':valid,
                                  'action':action,'feedback':int(positive) | (int(negative)<<1),
                                  'next':next_obs,'terminal_success':int(getattr(self,'captured',False))})
        self.captured = False

    def replay_experience(self, events):
        """Learn from recorded teachers without overwriting personal memory."""
        from .distillation import distill_into
        return distill_into(self,events)

    def grow(self, seed, max_neurons=Limits.DEFAULT_MAX_NEURONS):
        if len(self.nodes) >= min(max_neurons,Limits.MAX_NEURONS) or (len(self.nodes)+1)*self.width>Limits.MAX_STATE_BITS: return False
        rng=random.Random(seed);limit=self.sensors+len(self.nodes)
        n=Neuron(rng.randrange(limit),rng.randrange(limit),rng.choice(('AND','OR','XOR','COPY','NOT')))
        self.nodes.append(n);self.rules.append(self._rule(n,rng))
        self._ancestor_cache = None
        # No immediate output rewiring: growth initially preserves behavior.
        self.reset();return True

    def recovery_mutate(self, seed, max_neurons=Limits.DEFAULT_MAX_NEURONS, exploratory=False,
                        allow_growth=True, allow_reactivate=True):
        rng=random.Random(seed)
        options=['wire','table','shift','hold','output','write']
        if allow_reactivate:options.append('reactivate')
        if allow_growth and len(self.nodes)<max_neurons:
            growth_weight=4 if exploratory else 2
            options += ['grow']*growth_weight + ['grow_output']*growth_weight
        mode=rng.choice(options)
        i=rng.randrange(len(self.nodes));rule=self.rules[i]
        if mode=='grow':self.grow(seed,max_neurons)
        elif mode=='grow_output':
            if self.grow(seed,max_neurons):
                index=len(self.nodes)-1
                output=rng.randrange(len(self.outputs))
                self.outputs[output]=index
                self.output_lanes[output]=rng.randrange(self.width)
        elif mode=='wire':setattr(self.nodes[i],rng.choice(('a','b')),rng.randrange(self.sensors+len(self.nodes)))
        elif mode=='table':rule['lut'][rng.randrange(4)] ^= 1 << rng.randrange(self.width)
        elif mode=='shift':rule[rng.choice(('shift_a','shift_b'))]=rng.randrange(self.width)
        elif mode=='hold':rule['hold']=rng.randrange(-1,self.sensors+len(self.nodes))
        elif mode=='write':rule['write'] ^= 1 << rng.randrange(self.width)
        elif mode=='reactivate':self.nodes[i].active=True
        else:
            j=rng.randrange(self.action_bits);self.outputs[j]=i;self.output_lanes[j]=rng.randrange(self.width)
        self._ancestor_cache = None
        self.clear_circuit_evidence()
        self.reset()
        return {'kind':'vector_'+mode,'neuron':i}

    def remove_neuron(self, index):
        """Physically remove a module and remap every edge, gate and output.

        This is a structural mutation, not a semantics-preserving optimization.
        Evolution evaluates its consequences before it can be taught to EVA.
        """
        if len(self.nodes) <= 1 or not 0 <= index < len(self.nodes):
            return False
        old_reference = self.sensors + index
        replacement = self.nodes[index].a
        if replacement == old_reference:
            replacement = self.nodes[index].b
        if replacement == old_reference:
            replacement = index % self.sensors
        def remap(source):
            if source == old_reference:
                source = replacement
            return source - int(source > old_reference)
        del self.nodes[index]
        del self.rules[index]
        for node, rule in zip(self.nodes, self.rules):
            node.a, node.b = remap(node.a), remap(node.b)
            if rule['hold'] >= 0:
                rule['hold'] = remap(rule['hold'])
        for slot, output in enumerate(self.outputs):
            if output == index:
                target = remap(replacement) - self.sensors
                self.outputs[slot] = target if target >= 0 else min(index, len(self.nodes)-1)
            else:
                self.outputs[slot] = output - int(output > index)
        self._ancestor_cache = None
        self.clear_circuit_evidence()
        self.reset()
        return True

    def vary_topology(self, seed, minimum, maximum, compact=False):
        """Symmetric bounded topology exploration, independent of generation age."""
        rng = random.Random(seed)
        before = len(self.nodes)
        dormant=sorted(set(range(before))-set(self._output_ancestors()))
        direction = -1 if compact and dormant else rng.choice((-1, 0, 1))
        if direction < 0 and before > minimum:
            self.remove_neuron(rng.choice(dormant) if dormant else rng.randrange(before))
        elif direction > 0 and (dormant or self.grow(seed, maximum)):
            target=rng.choice(dormant) if dormant else len(self.nodes)-1
            # Connect the new module so extra capacity can affect behavior now.
            slot = rng.randrange(self.action_bits)
            self.outputs[slot] = target
            self.output_lanes[slot] = rng.randrange(self.width)
            self._ancestor_cache = None
            self.reset()
        return {'neurons_before': before, 'neurons_after': len(self.nodes),
                'topology_delta': len(self.nodes)-before}

    def rebuild_associative(self,mask=None):
        old=self.associative_memory
        self.associative_memory=AssociativeMemory(self.context_codec,old.capacity,old.width,
                                                 old.sensor_mask if mask is None else mask)
        for key in list(self.contextual_memory.rows)[-old.capacity:]:
            self.sync_associative(key)

    def change_context_history(self,depth):
        if depth==self.context_codec.history:return
        old_codec=self.context_codec
        codec=ContextCodec(self.sensors,depth,self.action_bits)
        old=self.contextual_memory;new=ContextMemory(old.capacity,self.actions)
        new.clock=old.clock
        for key in old.rows:
            old._expire(key)
            current,history=old_codec.decode(key)
            target=codec.encode(current,history)
            new._ensure(target)
            for source,bank in ((old.rows,new.rows),(old.terminal,new.terminal),(old.teachers,new.teachers)):
                if key in source:
                    bank[target]=[add(a,b,self.action_mask) for a,b in zip(bank.get(target,new.empty()),source[key])]
            meta=new.meta[target];before=old.meta[key]
            meta['updates']+=before['updates']
            meta['sources']=sorted(set(meta['sources'])|set(before['sources']))[:Limits.MEMORY_SOURCE_LIMIT]
            meta['last']=before['last']
            new.rows.move_to_end(target)
        self.context_codec,self.contextual_memory=codec,new
        self.rebuild_associative()
        self.reset()

    def mutate_representation(self,seed,max_width=Limits.MAX_EVOLVED_WIDTH):
        rng=random.Random(seed)
        i=rng.randrange(len(self.nodes));rule=self.rules[i];side=rng.choice(('a','b'))
        kind=rng.choice(('operator','axis','shape','delay','history','attention','width'))
        if kind=='operator':rule['operator_'+side]=rng.choice(OPERATORS)
        elif kind=='axis':rule['axis_'+side]=rng.randrange(len(rule['shape']))
        elif kind=='shape':rule['shape']=list(mutate_shape(rule['shape'],rng))
        elif kind=='delay':rule['delay_'+side]=rng.randrange(Limits.MAX_DELAY+1)
        elif kind=='history':self.change_context_history(rng.choice(Limits.HISTORY_CHOICES))
        elif kind=='attention':
            mask=self.associative_memory.sensor_mask ^ (1<<rng.randrange(self.sensors))
            if mask:self.rebuild_associative(mask)
        elif self.width*2<=max_width and self.width*2*len(self.nodes)<=Limits.MAX_STATE_BITS:
            self.extend_width((2,*self.shape) if len(self.shape)<Limits.MAX_AXES else (self.width*2,))
        self.clear_circuit_evidence();self.invalidate();self.reset()
        return {'kind':kind,'node':i,'width':self.width,'history':self.context_codec.history}

    def reshape(self, shape):
        shape=validate_shape(shape)
        if math.prod(shape)!=self.width:raise ValueError('Reshape preserves bit count')
        self.shape=shape
        for rule in self.rules:rule['shape']=list(shape)
        self.reset()

    def extend_width(self, shape):
        shape=validate_shape(shape);width=math.prod(shape)
        if width<self.width or width*len(self.nodes)>Limits.MAX_STATE_BITS:
            raise ValueError('Width/state budget exceeded')
        self.shape,self.width,self.mask=shape,width,(1<<width)-1
        for rule in self.rules:rule['shape']=list(shape)
        self.reset()

    def export(self):
        return {'engine':ENGINE,'sensors':self.sensors,'action_bits':self.action_bits,'nodes':[asdict(n) for n in self.nodes],
                'outputs':list(self.outputs),'state':list(self.state),
                'shape':list(self.shape),'sensor_encoding':self.sensor_encoding,'rules':[{**r,'lut':list(r['lut']),'shape':list(r.get('shape',self.shape))} for r in self.rules],
                'output_lanes':list(self.output_lanes),'memory_capacity':self.memory_capacity,
                'internal_ticks':self.internal_ticks,'trace_depth':self.trace_depth,
                'knowledge':self.short_memory.export(),
                'contextual_memory':self.contextual_memory.export(),
                'context_history_depth':self.context_codec.history,
                'associative_memory':self.associative_memory.export(),
                'circuit_evidence':[[list(k),list(v)] for k,v in sorted(self.circuit_evidence.items())],
                'predictions':list(self.predictions.items()),
                'vector_runtime':{'vectors':list(self.vectors),'context':self.context,
                                  'context_valid':self.context_valid,'trace':list(self.trace),
                                  'last':self.last,'updates':self.update_count,
                                  'observable_history':list(self.observable_history),
                                  'context_trace':list(self.context_trace),
                                  'current_context_key':self.current_context_key,
                                  'input_history':list(self.input_history),'circuit_trace':list(self.circuit_trace),
                                  'last_payload':self.last_payload,'next_observation':self.next_observation,
                                  'captured':self.captured}}

    @classmethod
    def restore(cls,data):
        if data.get('engine')!=ENGINE:
            raise ValueError('Formato di un’altra versione: avviare un nuovo profilo V6.05')
        obj=cls(sensors=data['sensors'],neurons=0,shape=data['shape'],
                memory_capacity=data['memory_capacity'],internal_ticks=data['internal_ticks'],
                trace_depth=data['trace_depth'], action_bits=data['action_bits'],
                context_history=data['context_history_depth'],hdc_width=data['associative_memory']['width'],
                hdc_capacity=data['associative_memory']['capacity'])
        obj.sensor_encoding=data.get('sensor_encoding','mixed')
        if obj.sensor_encoding not in ('mixed','broadcast'):raise ValueError('Unknown sensor encoding')
        obj.nodes=[Neuron(**n) for n in data['nodes']]
        obj.outputs=list(data['outputs']);obj.output_lanes=list(data['output_lanes'])
        obj.rules=[{**r,'lut':list(r['lut'])} for r in data['rules']]
        if (len(obj.rules)!=len(obj.nodes) or len(obj.outputs)!=obj.action_bits or len(obj.output_lanes)!=obj.action_bits or
                len(obj.nodes)>Limits.MAX_NEURONS or len(obj.nodes)*obj.width>Limits.MAX_STATE_BITS):
            raise ValueError('Invalid vector circuit')
        for n,r in zip(obj.nodes,obj.rules):
            if not 0<=n.a<obj.sensors+len(obj.nodes) or not 0<=n.b<obj.sensors+len(obj.nodes):
                raise ValueError('Invalid connection')
            if not -1<=r['hold']<obj.sensors+len(obj.nodes) or len(r['lut'])!=4:
                raise ValueError('Invalid hold/table')
            if math.prod(validate_shape(r['shape']))!=obj.width:raise ValueError('Invalid local shape')
            for side in ('a','b'):
                if r.get('operator_'+side,'rotate') not in OPERATORS or not 0<=r.get('delay_'+side,0)<=Limits.MAX_DELAY:
                    raise ValueError('Invalid tensor operator or delay')
            if any(type(v) is not int or v<0 or v>obj.mask for v in r['lut']+[r['write']]):
                raise ValueError('Table outside Boolean width')
        if any(not 0<=i<len(obj.nodes) for i in obj.outputs) or any(not 0<=i<obj.width for i in obj.output_lanes):
            raise ValueError('Invalid output')
        obj.contextual_memory.restore(data['contextual_memory'],obj.context_codec)
        obj.short_memory.restore(data['knowledge']);obj.knowledge=obj.short_memory.rows
        a=data['associative_memory']
        obj.associative_memory=AssociativeMemory(obj.context_codec,a['capacity'],a['width'],int(a['sensor_mask']))
        obj.associative_memory.restore(a)
        obj.predictions=OrderedDict(data.get('predictions',[]))
        if len(obj.predictions)>obj.memory_capacity:raise ValueError('Prediction capacity exceeded')
        obj.circuit_evidence={tuple(k):list(v) for k,v in data['circuit_evidence']}
        obj.reset();runtime=data.get('vector_runtime',{})
        if runtime:
            obj.vectors=list(runtime['vectors']);obj.context=runtime['context']
            obj.context_valid=runtime['context_valid'];obj.trace=deque(runtime.get('trace',[]),maxlen=obj.trace_depth)
            obj.last=runtime.get('last');obj.update_count=runtime.get('updates',0)
            obj.observable_history=deque(runtime.get('observable_history',[]),maxlen=Limits.MAX_CONTEXT_HISTORY)
            obj.context_trace=deque(runtime.get('context_trace',[]),maxlen=obj.trace_depth)
            obj.current_context_key=runtime.get('current_context_key')
            obj.input_history=deque(runtime['input_history'],maxlen=Limits.MAX_DELAY)
            obj.circuit_trace=deque(runtime['circuit_trace'],maxlen=obj.trace_depth)
            obj.last_payload=runtime['last_payload'];obj.next_observation=runtime['next_observation']
            obj.captured=runtime['captured']
            if any(len(row)!=obj.sensors+len(obj.nodes) or any(not 0<=v<=obj.mask for v in row)
                   for row in obj.input_history):raise ValueError('Invalid delay state')
            if len(obj.vectors)!=len(obj.nodes) or any(not 0<=v<=obj.mask for v in obj.vectors):
                raise ValueError('Invalid vector state')
            obj.state=[v&1 for v in obj.vectors]
        return obj

    def snapshot_id(self):
        d=self.export();d.pop('state');d.pop('vector_runtime')
        return hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()[:20]

    def frame_data(self):
        return {'vector_state':[hex(v) for v in self.vectors],
                'vector_previous':[hex(v) for v in self.before_vectors],
                'vector_width':self.width,'vector_shape':list(self.shape),
                'vector_updates':list(self.last_updates),'vector_update_count':self.update_count,
                'knowledge_entries':len(self.knowledge),'contextual_entries':len(self.contextual_memory.rows),'decoder':dict(self.last_decoder),
                'prediction_error':self.last_prediction_error,'computed_nodes':list(self.computed_nodes),
                'executed_neurons':len(self.computed_nodes),'decoder_counts':dict(self.decoder_counts),
                'associative_entries':len(self.associative_memory.entries)}
