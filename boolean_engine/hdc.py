"""Binary VSA coding and bounded, conservative associative retrieval."""
from collections import OrderedDict, Counter
from functools import lru_cache
import hashlib
from .limits import CoreLimits as L
from .tensor import rotate
from .context import ContextCodec


def majority(vectors, width, tie=0):
    vectors = tuple(vectors)
    if not vectors: return tie & ((1 << width)-1)
    planes = []
    for vector in vectors:
        carry = vector & ((1 << width)-1)
        for i,old in enumerate(planes):
            planes[i],carry = old ^ carry, old & carry
            if not carry: break
        else:
            if carry: planes.append(carry)
    equal = (1 << width)-1
    greater = 0
    threshold = len(vectors)//2
    for i in reversed(range(max(len(planes),threshold.bit_length()))):
        plane = planes[i] if i<len(planes) else 0
        if (threshold>>i)&1: equal &= plane
        else:
            greater |= equal & plane
            equal &= ~plane
    return greater | (equal & tie if len(vectors)%2==0 else 0)


class BinarySpace:
    """Symbol IDs are reproducible; similarity comes from shared features."""
    def __init__(self, width=L.HDC_WIDTH, seed=0, roles=None):
        if not L.MIN_WIDTH <= width <= L.HDC_MAX_WIDTH:
            raise ValueError('Invalid vector width')
        self.width,self.seed = width,seed
        self.mask = (1 << width)-1
        self.roles = dict(roles or {})

    def symbol(self, name):
        return _symbol(self.width,self.seed,name)

    def register(self, name):
        if name not in self.roles: self.roles[name]=self.symbol('role:'+name)
        return self.roles[name]

    def scalar(self, role, level, levels):
        if not 1 <= levels <= self.width or not 0 <= level <= levels:
            raise ValueError('Invalid scalar code')
        return self.register(role) ^ ((1 << level)-1)

    def bind(self, role, symbol):
        return self.register(role) ^ self.symbol('symbol:'+symbol)

    def bundle(self, vectors):
        return majority(vectors,self.width,self.symbol('tie'))

    def permute(self, vector, position):
        return rotate(vector,position,self.width)

    @staticmethod
    def distance(a,b): return (a^b).bit_count()

    def export(self): return {'width':self.width,'seed':self.seed,'roles':dict(self.roles)}


class AssociativeMemory:
    """Hamming neighbours must agree; never use hidden environment information."""
    def __init__(self, codec, capacity=L.HDC_CAPACITY, width=L.HDC_WIDTH, sensor_mask=None):
        if not L.HDC_MIN_WIDTH <= width <= L.HDC_MAX_WIDTH or capacity<1:
            raise ValueError('Invalid associative budget')
        self.codec,self.capacity,self.width = codec,capacity,width
        self.space=BinarySpace(width)
        self.sensor_mask=(1 << codec.sensors)-1 if sensor_mask is None else sensor_mask
        if not 0 <= self.sensor_mask < (1 << codec.sensors): raise ValueError('Invalid sensor mask')
        self.entries=OrderedDict()
        self.buckets={}
        self.last_lookup={}

    def encode(self,key):
        return _encode(self.codec.sensors,self.codec.history,self.codec.action_bits,self.sensor_mask,self.width,key)

    def _remove(self,key):
        old=self.entries.pop(key,None)
        if old is not None:
            bucket=self.buckets[old[1]]
            bucket.pop(key,None)
            if not bucket:self.buckets.pop(old[1])

    def observe(self,key,positive,negative):
        if not positive and not negative:
            self._remove(key);return
        vector=self.encode(key)
        anchor=self.codec.decode(key)[0] & self.sensor_mask
        self._remove(key)
        self.entries[key]=(vector,anchor,positive,negative)
        self.buckets.setdefault(anchor,OrderedDict())[key]=None
        if len(self.entries)>self.capacity:self._remove(next(iter(self.entries)))

    def choose(self,key,proposed,rejected=0,allowed=None):
        current,_=self.codec.decode(key)
        bucket=self.buckets.get(current & self.sensor_mask,{})
        keys=list(bucket)[-L.HDC_CANDIDATES:]
        if len(keys)<L.HDC_MIN_CONSENSUS:return None
        query=self.encode(key)
        nearest=sorted(((query^self.entries[k][0]).bit_count(),k)
                       for k in keys if k!=key and (allowed is None or k in allowed))
        nearest=[pair for pair in nearest if pair[0]*L.HDC_DISTANCE_DENOMINATOR<=self.width][:L.HDC_NEIGHBORS]
        positive=Counter();negative=Counter()
        for distance,k in nearest:
            for action in range(self.codec.actions):
                positive[action]+=bool(self.entries[k][2] & (1<<action))
                negative[action]+=bool(self.entries[k][3] & (1<<action))
        candidates=[a for a in range(self.codec.actions) if not rejected & (1<<a)
                    and positive[a]>=L.HDC_MIN_CONSENSUS and positive[a]>negative[a]]
        ranked=sorted(candidates,key=lambda a:(-(positive[a]-negative[a]),(a-proposed)%self.codec.actions))
        result=None
        if ranked:
            best=ranked[0]
            rival=max((positive[a]-negative[a] for a in range(self.codec.actions) if a!=best),default=0)
            if positive[best]-negative[best]>=rival+L.HDC_VOTE_MARGIN:result=best
        self.last_lookup={'neighbors':len(nearest),'action':result,
                          'distances':[d for d,_ in nearest]}
        return result

    def export(self):
        return {'width':self.width,'capacity':self.capacity,'sensor_mask':str(self.sensor_mask),
                'entries':[[k,hex(v),str(a),p,n] for k,(v,a,p,n) in self.entries.items()]}

    def restore(self,data):
        if data['width']!=self.width or data['capacity']!=self.capacity or int(data['sensor_mask'])!=self.sensor_mask:
            raise ValueError('Associative schema mismatch')
        if len(data['entries'])>self.capacity:raise ValueError('Associative capacity exceeded')
        for key,vector,anchor,pos,neg in data['entries']:
            if not self.codec.validate(key) or not 0<=pos<=self.codec.action_mask or not 0<=neg<=self.codec.action_mask:
                raise ValueError('Invalid associative entry')
            if int(vector,16)!=self.encode(key) or int(anchor)!=(self.codec.decode(key)[0]&self.sensor_mask):
                raise ValueError('Associative code inconsistent with context')
            self.observe(key,pos,neg)


@lru_cache(maxsize=L.CONTEXT_CACHE_SIZE)
def _symbol(width,seed,name):
    raw=hashlib.shake_256(f'{seed}:{name}'.encode()).digest((width+7)//8)
    return int.from_bytes(raw,'little') & ((1<<width)-1)


@lru_cache(maxsize=L.CONTEXT_CACHE_SIZE)
def _encode(sensors,depth,action_bits,sensor_mask,width,key):
    codec=ContextCodec(sensors,depth,action_bits)
    space=BinarySpace(width)
    current,history=codec.decode(key)
    parts=[]
    for lag,(observation,action) in enumerate([(current,None)]+list(reversed(history))):
        for bit in range(codec.sensors):
            if sensor_mask & (1<<bit):
                feature=space.bind(f'feature:{bit}',str((observation>>bit)&1))
                parts.append(space.permute(feature,lag*31))
        if action is not None:
            parts.append(space.permute(space.bind('action',str(action)),lag*31))
    parts.append(space.bind('history_length',str(len(history))))
    return space.bundle(parts)
