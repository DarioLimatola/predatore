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

from .boolean_net import BooleanNet, Neuron, gate

from .settings import Limits
from .contextual import ContextCodec, ContextMemory

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


class BinarySpace:
    """Versioned role/value encoding. Symbols are identities, not learned semantics.

    Numeric levels use thermometer coding, preserving Hamming distance locally.
    Adding roles never renames old roles. Superposition has finite capacity.
    """
    def __init__(self, width=Limits.DEFAULT_WIDTH, seed=0, roles=None):
        if not Limits.MIN_WIDTH <= width <= Limits.MAX_WIDTH:
            raise ValueError('width must be 8..4096')
        self.width, self.seed = width, seed
        self.roles = dict(roles or {})

    def symbol(self, name):
        raw = hashlib.shake_256(f'{self.seed}:{name}'.encode()).digest((self.width+7)//8)
        return int.from_bytes(raw, 'little') & ((1 << self.width)-1)

    def register(self, name):
        if name not in self.roles:
            self.roles[name] = self.symbol('role:'+name)
        return self.roles[name]

    def scalar(self, role, level, levels):
        if not 1 <= levels <= self.width or not 0 <= level <= levels:
            raise ValueError('Scalar outside declared quantized range')
        return self.register(role) ^ ((1 << level)-1)

    def bind(self, role, symbol):
        return self.register(role) ^ self.symbol('symbol:'+symbol)

    def export(self):
        return {'width':self.width,'seed':self.seed,'roles':dict(self.roles)}


class VectorBooleanNet(BooleanNet):
    """Sparse graph of n-dimensional bit modules, synchronous bounded ticks.

    RAM evidence is part of the neural decoder, not an external movement rule.
    Persistent evidence and circuits survive reset(); working state does not.
    """
    def __init__(self, seed=1, sensors=Limits.SENSOR_BITS, neurons=Limits.DEFAULT_BASE_NEURONS, shape=Limits.DEFAULT_SHAPE,
                 memory_capacity=Limits.DEFAULT_MEMORY, internal_ticks=Limits.DEFAULT_TICKS, trace_depth=Limits.DEFAULT_TRACE):
        width = math.prod(shape)
        if not shape or any(type(d) is not int or d <= 0 for d in shape) or not Limits.MIN_WIDTH <= width <= Limits.MAX_WIDTH:
            raise ValueError('Invalid module shape (8..4096 bits)')
        if not 1 <= internal_ticks <= Limits.MAX_TICKS or not 1 <= trace_depth <= Limits.MAX_TRACE or memory_capacity < 1:
            raise ValueError('Invalid resource budget')
        super().__init__(seed,sensors,neurons)
        self.shape = tuple(shape); self.width = width; self.mask = (1 << width)-1
        self.memory_capacity = memory_capacity; self.internal_ticks = internal_ticks
        self.trace_depth = trace_depth
        rng = random.Random(seed ^ 0xB17)
        self.rules = [self._rule(n, rng) for n in self.nodes]
        self.output_lanes = [rng.randrange(width) for _ in range(Limits.ACTION_BITS)]
        self.sensor_encoding = "mixed"
        self.knowledge = OrderedDict()
        self.predictions = OrderedDict()
        self.context_codec = ContextCodec(sensors)
        self.contextual_memory = ContextMemory(memory_capacity)
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
                'write':self.mask,
                'lut':[self.mask if gate(node.op,a,b) else 0 for a,b in ((0,0),(0,1),(1,0),(1,1))]}

    def reset(self):
        self.vectors = [0]*len(self.nodes)
        self.state = [0]*len(self.nodes) # Lane-zero projection for displays.
        self.context = 0; self.context_valid = 0
        self.observable_history = deque(maxlen=Limits.CONTEXT_HISTORY)
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

    def _address(self, observation):
        current = pack(observation)
        # Validity distinguishes initial context from an actual all-zero observation.
        return current | (self.context << self.sensors) | (self.context_valid << (2*self.sensors))

    def decide(self, observation):
        if len(observation) != self.sensors:
            raise ValueError('sensor width mismatch')
        obs = pack(observation)
        address = obs | (self.context << self.sensors) | (self.context_valid << (2*self.sensors))
        self.current_context_key = self.context_codec.encode(obs,self.observable_history if self.memory_enabled else ())
        self.before_vectors = list(self.vectors)
        activity = []
        collect_activity = self.collect_experience and self.circuit_plasticity_enabled
        activity_nodes = self._output_ancestors() if collect_activity else ()
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
            new = []
            tick_activity = {} if collect_activity else None
            for i,(node,r) in enumerate(zip(nodes,rules)):
                if not node.active:
                    new.append(0)
                    continue
                shift_a = r['shift_a'] % width
                shift_b = r['shift_b'] % width
                source_a = inputs[node.a]
                source_b = inputs[node.b]
                a = source_a if not shift_a else (
                    (source_a << shift_a) | (source_a >> (width-shift_a))) & mask
                b = source_b if not shift_b else (
                    (source_b << shift_b) | (source_b >> (width-shift_b))) & mask
                na,nb = a ^ mask,b ^ mask
                t = r['lut']
                hold = inputs[r['hold']] if r['hold'] >= 0 and self.memory_enabled else 0
                write = r['write'] & ~hold & mask
                if collect_activity and i in activity_nodes:
                    tick_activity[i] = (na & nb & write, na & b & write,
                                        a & nb & write, a & b & write)
                value = ((na & nb & t[0]) | (na & b & t[1]) |
                         (a & nb & t[2]) | (a & b & t[3]))
                new.append(((old[i] & ~write) | (value & write)) & mask)
            if collect_activity:
                activity = tick_activity
            self.vectors = new
        self.last_activity = activity if collect_activity else []
        self.state = [v & 1 for v in self.vectors]
        proposed = sum(((self.vectors[n] >> lane) & 1) << bit
                       for bit,(n,lane) in enumerate(zip(outputs,output_lanes)))
        low,high = self.knowledge.get(str(address),(Limits.ACTION_MASK,0))
        good = low & high
        bad = ~(low | high) & Limits.ACTION_MASK
        choices = good if good else (~bad & Limits.ACTION_MASK)
        action = proposed
        if self.associative_enabled and choices:
            # Boolean priority multiplexer, ordered by recurrent circuit output.
            for offset in range(Limits.ACTIONS):
                candidate = (proposed+offset) & (Limits.ACTIONS-1)
                if choices & (1 << candidate):
                    action = candidate; break
        contextual_action = (self.contextual_memory.choose(self.current_context_key,action)
                             if self.contextual_enabled and self.associative_enabled else None)
        if contextual_action is not None:action=contextual_action
        self.last_decoder = {'context_key':self.current_context_key,'context_override':contextual_action,
                             'proposed':proposed,'good_mask':good,'bad_mask':bad,
                             'address':str(address),'action':action}
        if collect_activity:
            ancestors = self._output_ancestors()
            active_patterns = []
            for node in sorted(ancestors):
                for pattern, lanes in enumerate(self.last_activity.get(node, ())):
                    lanes &= self.rules[node]['write']
                    if lanes:
                        active_patterns.append((node, pattern, lanes))
            self.circuit_trace.append((action == proposed, active_patterns))
        self.context_trace.append((self.current_context_key,action))
        if self.memory_enabled:self.observable_history.append((obs,action))
        else:self.observable_history.clear()
        self.last = (address,action,obs,self.context,self.context_valid)
        self.trace.append(self.last)
        self.context = obs if self.memory_enabled else 0
        self.context_valid = int(self.memory_enabled)
        return action

    def learn(self, address, action, positive):
        """Saturating 2-bit evidence per action, updated with Boolean bit planes.

        00 reject, 01/10 uncertain, 11 support. Repetition reinforces, not XOR-cancels.
        """
        key = str(address); low,high = self.knowledge.get(key,(Limits.ACTION_MASK,0))
        bit = 1 << action
        if positive:
            write = bit & ~(low & high)
            nl,nh = low ^ write, high ^ (low & write)
        else:
            write = bit & (low | high)
            nl,nh = low ^ write, high ^ (~low & write)
        self.knowledge[key] = [nl & Limits.ACTION_MASK,nh & Limits.ACTION_MASK]
        self.knowledge.move_to_end(key)
        if len(self.knowledge) > self.memory_capacity:
            self.knowledge.popitem(last=False)
        if (nl,nh) != (low,high):
            self.update_count += 1
            self.last_updates.append({'address':key,'action':action,'before':[low,high],
                                      'after':[nl & Limits.ACTION_MASK,nh & Limits.ACTION_MASK]})

    def transition(self, next_observation, captured=False):
        self.next_observation = pack(next_observation)
        self.captured = bool(captured)

    def clear_circuit_evidence(self):
        self.circuit_evidence.clear()
        self.circuit_trace.clear()

    def _output_ancestors(self):
        if self._ancestor_cache is not None:
            return self._ancestor_cache
        ancestors = set(self.outputs)
        pending = list(ancestors)
        while pending:
            node_index = pending.pop()
            node = self.nodes[node_index]
            for source in (node.a, node.b, self.rules[node_index]['hold']):
                if source >= self.sensors:
                    upstream = source - self.sensors
                    if upstream not in ancestors:
                        ancestors.add(upstream)
                        pending.append(upstream)
        self._ancestor_cache = ancestors
        return ancestors

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
        if (self.circuit_plasticity_enabled and reward != 0):
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
            # Bounded terminal-success eligibility trace. It is experimental credit
            # assignment, not a claim that every preceding action caused the capture.
            if getattr(self,'captured',False):
                for key,act in self.context_trace:
                    self.contextual_memory.update(key,act,True,Limits.CONTEXT_TERMINAL_CREDIT)
                for a,act,*_ in list(self.trace)[:-1]:
                    self.learn(a,act,True)
            self.predictions[prediction_key] = next_obs
            self.predictions.move_to_end(prediction_key)
            if len(self.predictions) > self.memory_capacity:
                self.predictions.popitem(last=False)
        if self.collect_experience and self.experience_sink is not None:
            self.experience_sink({'context_key':self.current_context_key,
                                  'eligible_contexts':list(self.context_trace) if self.captured else [],
                                  'observation':obs,'previous':context,'valid':valid,
                                  'action':action,'feedback':int(positive) | (int(negative)<<1),
                                  'next':next_obs,'terminal_success':int(getattr(self,'captured',False))})
        self.captured = False

    def replay_experience(self, events):
        """Shared fixed observation schema permits cross-agent RAM learning."""
        for e in events:
            if e['feedback'] not in (1,2): continue
            address = e['observation'] | (e['previous'] << self.sensors) | (e['valid'] << (2*self.sensors))
            self.learn(address,e['action'],e['feedback']==1)
            key=f'{address}:{e["action"]}'
            self.predictions[key]=e['next']
            self.predictions.move_to_end(key)
            if len(self.predictions)>self.memory_capacity:self.predictions.popitem(last=False)
        self.reset()

    def grow(self, seed, max_neurons=Limits.DEFAULT_MAX_NEURONS):
        if len(self.nodes) >= max_neurons: return False
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
            j=rng.randrange(Limits.ACTION_BITS);self.outputs[j]=i;self.output_lanes[j]=rng.randrange(self.width)
        self._ancestor_cache = None
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
        self.reset()
        return True

    def vary_topology(self, seed, minimum, maximum):
        """Symmetric bounded topology exploration, independent of generation age."""
        rng = random.Random(seed)
        before = len(self.nodes)
        direction = rng.choice((-1, 0, 1))
        if direction < 0 and before > minimum:
            self.remove_neuron(rng.randrange(before))
        elif direction > 0 and before < maximum:
            self.grow(seed, maximum)
            # Connect the new module so extra capacity can affect behavior now.
            slot = rng.randrange(Limits.ACTION_BITS)
            self.outputs[slot] = len(self.nodes)-1
            self.output_lanes[slot] = rng.randrange(self.width)
            self._ancestor_cache = None
            self.reset()
        return {'neurons_before': before, 'neurons_after': len(self.nodes),
                'topology_delta': len(self.nodes)-before}

    def reshape(self, shape):
        if not shape or any(type(d) is not int or d<1 for d in shape) or math.prod(shape)!=self.width:
            raise ValueError('Reshape preserves bit count; use extend_width to grow')
        self.shape=tuple(shape)

    def extend_width(self, shape):
        width=math.prod(shape)
        if width<self.width or width>Limits.MAX_WIDTH or any(type(d) is not int or d<1 for d in shape):
            raise ValueError('Width may grow up to 4096')
        # Existing masks and tables occupy the old lanes. New lanes start at zero.
        # Rotation semantics can change: require behavioral promotion after expansion.
        self.shape=tuple(shape);self.width=width;self.mask=(1<<width)-1;self.reset()

    def export(self):
        return {'engine':ENGINE,'sensors':self.sensors,'nodes':[asdict(n) for n in self.nodes],
                'outputs':list(self.outputs),'state':list(self.state),
                'shape':list(self.shape),'sensor_encoding':self.sensor_encoding,'rules':[{**r,'lut':list(r['lut'])} for r in self.rules],
                'output_lanes':list(self.output_lanes),'memory_capacity':self.memory_capacity,
                'internal_ticks':self.internal_ticks,'trace_depth':self.trace_depth,
                'knowledge':[[k,list(v)] for k,v in self.knowledge.items()],
                'contextual_memory':self.contextual_memory.export(),
                'context_history_depth':Limits.CONTEXT_HISTORY,
                'predictions':list(self.predictions.items()),
                'vector_runtime':{'vectors':list(self.vectors),'context':self.context,
                                  'context_valid':self.context_valid,'trace':list(self.trace),
                                  'last':self.last,'updates':self.update_count,
                                  'observable_history':list(self.observable_history),
                                  'context_trace':list(self.context_trace),
                                  'current_context_key':self.current_context_key}}

    @classmethod
    def restore(cls,data):
        if data.get('engine')!=ENGINE:
            raise ValueError('Formato di un’altra versione: avviare un nuovo profilo V6.04')
        obj=cls(sensors=data['sensors'],neurons=0,shape=data['shape'],
                memory_capacity=data['memory_capacity'],internal_ticks=data['internal_ticks'],
                trace_depth=data['trace_depth'])
        obj.sensor_encoding=data.get('sensor_encoding','mixed')
        if obj.sensor_encoding not in ('mixed','broadcast'):raise ValueError('Unknown sensor encoding')
        obj.nodes=[Neuron(**n) for n in data['nodes']]
        obj.outputs=list(data['outputs']);obj.output_lanes=list(data['output_lanes'])
        obj.rules=[{**r,'lut':list(r['lut'])} for r in data['rules']]
        if len(obj.rules)!=len(obj.nodes) or len(obj.outputs)!=Limits.ACTION_BITS or len(obj.output_lanes)!=Limits.ACTION_BITS:
            raise ValueError('Invalid vector circuit')
        for n,r in zip(obj.nodes,obj.rules):
            if not 0<=n.a<obj.sensors+len(obj.nodes) or not 0<=n.b<obj.sensors+len(obj.nodes):
                raise ValueError('Invalid connection')
            if not -1<=r['hold']<obj.sensors+len(obj.nodes) or len(r['lut'])!=4:
                raise ValueError('Invalid hold/table')
            if any(type(v) is not int or v<0 or v>obj.mask for v in r['lut']+[r['write']]):
                raise ValueError('Table outside Boolean width')
        if any(not 0<=i<len(obj.nodes) for i in obj.outputs) or any(not 0<=i<obj.width for i in obj.output_lanes):
            raise ValueError('Invalid output')
        if data.get('context_history_depth')!=Limits.CONTEXT_HISTORY:
            raise ValueError('Observable history schema mismatch')
        obj.contextual_memory.restore(data.get('contextual_memory',[]),obj.context_codec)
        obj.knowledge=OrderedDict((str(k),list(v)) for k,v in data.get('knowledge',[]))
        obj.predictions=OrderedDict(data.get('predictions',[]))
        if len(obj.knowledge)>obj.memory_capacity or len(obj.predictions)>obj.memory_capacity:
            raise ValueError('Memory budget exceeded')
        for k,v in obj.knowledge.items():
            if int(k)<0 or len(v)!=2 or any(type(x) is not int or not 0<=x<=Limits.ACTION_MASK for x in v):
                raise ValueError('Invalid associative memory')
        obj.reset();runtime=data.get('vector_runtime',{})
        if runtime:
            obj.vectors=list(runtime['vectors']);obj.context=runtime['context']
            obj.context_valid=runtime['context_valid'];obj.trace=deque(runtime.get('trace',[]),maxlen=obj.trace_depth)
            obj.last=runtime.get('last');obj.update_count=runtime.get('updates',0)
            obj.observable_history=deque(runtime.get('observable_history',[]),maxlen=Limits.CONTEXT_HISTORY)
            obj.context_trace=deque(runtime.get('context_trace',[]),maxlen=obj.trace_depth)
            obj.current_context_key=runtime.get('current_context_key')
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
                'prediction_error':self.last_prediction_error}
