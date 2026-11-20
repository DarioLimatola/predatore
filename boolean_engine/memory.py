"""Finite Boolean evidence RAM; one bit plane contains all actions."""
from collections import OrderedDict
from .limits import CoreLimits as L


def increment(planes, lanes, mask=L.ACTION_MASK):
    full = mask
    for plane in planes:
        full &= plane
    carry = lanes & ~full & mask
    for i, old in enumerate(planes):
        if not carry:
            break
        planes[i], carry = old ^ carry, old & carry


def greater(a, b, mask=L.ACTION_MASK):
    equal, result = mask, 0
    for x,y in zip(reversed(a),reversed(b)):
        result |= equal & x & ~y
        equal &= ~(x ^ y)
    return result & mask


def difference(a, b, mask=L.ACTION_MASK):
    borrow, result = 0, []
    for x,y in zip(a,b):
        result.append((x ^ y ^ borrow) & mask)
        borrow = ((~x & (y | borrow)) | (y & borrow)) & mask
    return result


def add(a, b, mask):
    carry, result = 0, []
    for x,y in zip(a,b):
        result.append((x^y^carry)&mask)
        carry = ((x&y)|(x&carry)|(y&carry))&mask
    return [x|carry for x in result]


def counter_value(planes, action):
    return sum(((plane>>action)&1)<<i for i,plane in enumerate(planes))


class ContextMemory:
    def __init__(self, capacity, actions=L.ACTIONS):
        if capacity < 1 or not 2 <= actions <= 1 << L.MAX_ACTION_BITS:
            raise ValueError('Invalid memory budget')
        self.capacity, self.actions, self.mask = capacity, actions, (1<<actions)-1
        self.minimum = [self.mask if L.CONTEXT_MIN_SUPPORT>>i & 1 else 0
                        for i in range(L.CONTEXT_COUNTER_BITS)]
        self.clear()

    def clear(self):
        self.rows=OrderedDict(); self.terminal={}; self.teachers={}; self.meta={}
        self.clock=0; self._cache={}; self._packed={}

    @staticmethod
    def empty():
        return [[0]*L.CONTEXT_COUNTER_BITS for _ in range(2)]

    @staticmethod
    def supported(row, mask=L.ACTION_MASK):
        positive,negative=row
        threshold=[mask if L.CONTEXT_MIN_SUPPORT>>i & 1 else 0 for i in range(len(positive))]
        return ~greater(threshold,positive,mask) & greater(positive,negative,mask)

    def _ensure(self,key):
        if key not in self.rows:
            self.rows[key]=self.empty()
            self.meta[key]={'last':self.clock,'decayed':self.clock,'updates':0,'sources':[],'signature':''}

    def _expire(self,key):
        meta=self.meta[key]
        n=min(L.CONTEXT_COUNTER_BITS,(self.clock-meta['decayed'])//L.MEMORY_DECAY_INTERVAL)
        if n:
            for bank in (self.rows,self.terminal,self.teachers):
                if key in bank:
                    bank[key]=[p[n:]+[0]*n for p in bank[key]]
            meta['decayed']=self.clock
            self._cache.pop(key,None); self._packed.pop(key,None)

    def _touch(self,key):
        self.meta[key]['last']=self.clock
        self.rows.move_to_end(key)
        self._cache.pop(key,None); self._packed.pop(key,None)
        if len(self.rows)>self.capacity:
            from itertools import islice
            old=min(islice(self.rows,L.MEMORY_EVICTION_WINDOW),key=lambda k:(
                bool(k in self.terminal and any(self.terminal[k][0])),
                bool(self.meta[k]['sources']),self.meta[k]['last']))
            for bank in (self.rows,self.terminal,self.teachers,self.meta,self._cache,self._packed):
                bank.pop(old,None)

    def update(self,key,action,positive,weight=1,*,channel='personal'):
        if not 0 <= action < self.actions or channel not in ('personal','terminal') or not 1 <= weight <= L.CONTEXT_TERMINAL_CREDIT:
            raise ValueError('Invalid evidence update')
        key=str(key)
        self.clock+=1; self._ensure(key); self._expire(key)
        row=self.rows[key] if channel=='personal' else self.terminal.setdefault(key,self.empty())
        for _ in range(weight):
            increment(row[0 if positive else 1],1<<action,self.mask)
        self.meta[key]['updates']+=1
        self._touch(key)

    def set_row(self,key,row):
        key=str(key); self._ensure(key)
        self.rows[key]=[list(p) for p in row]; self._touch(key)

    def set_teacher(self,key,row,sources,signature):
        self._ensure(key)
        if self.meta[key]['signature']==signature:
            return False
        self._expire(key)
        self.teachers[key]=[list(p) for p in row]
        self.meta[key].update(sources=sorted(set(sources))[:L.MEMORY_SOURCE_LIMIT],signature=signature)
        self._touch(key)
        return True

    def evidence(self,key):
        if key not in self.rows:
            return None
        self._expire(key)
        if key not in self._cache:
            row=self.rows[key]
            for bank in (self.terminal,self.teachers):
                if key in bank:
                    row=[add(a,b,self.mask) for a,b in zip(row,bank[key])]
            pos,neg=row
            good=~greater(self.minimum,pos,self.mask) & greater(pos,neg,self.mask)
            bad=~greater(self.minimum,neg,self.mask) & greater(neg,pos,self.mask)
            self._cache[key]=(good,bad,difference(pos,neg,self.mask) if good else ())
        return self._cache[key]

    def choose(self,key,proposed):
        found=self.evidence(str(key))
        if found is None:
            return None
        winners,rejected,margin=found
        if not winners:
            if not rejected & (1<<proposed):
                return None
            winners=~rejected & self.mask
        for plane in reversed(margin):
            winners=(winners & plane) or winners
        for offset in range(self.actions):
            action=(proposed+offset)%self.actions
            if winners>>action & 1:
                return action
        return None

    def export(self):
        size=(self.actions+7)//8
        if size==1:
            # Same bytes as the generic path: one little-endian byte per plane value.
            def pack(row):
                return bytes([v for p in row for v in p]).hex()
        else:
            def pack(row):
                return b''.join(v.to_bytes(size,'little') for p in row for v in p).hex()
        # Packed rows are cached; every write path already invalidates _cache for the same key.
        packed=self._packed
        def cached(k,row):
            hit=packed.get(k)
            if hit is None:
                hit=packed[k]=(pack(row),pack(self.terminal[k]) if k in self.terminal else None,
                               pack(self.teachers[k]) if k in self.teachers else None)
            return hit
        return {'clock':self.clock,'rows':[[k,*cached(k,row),dict(self.meta[k])]
                 for k,row in self.rows.items()]}

    def digest(self):
        import hashlib, json
        return hashlib.sha256(json.dumps(self.export(),sort_keys=True).encode()).hexdigest()

    def copy(self):
        import copy as _copy
        return _copy.deepcopy(self)

    def restore(self,data,codec=None):
        self.clear()
        self.clock=data['clock']
        if type(self.clock) is not int or self.clock<0 or len(data['rows'])>self.capacity:
            raise ValueError('Invalid memory budget')
        size=(self.actions+7)//8
        bits=L.CONTEXT_COUNTER_BITS
        def unpack(value):
            raw=bytes.fromhex(value)
            vals=list(raw) if size==1 else [int.from_bytes(raw[i:i+size],'little') for i in range(0,len(raw),size)]
            if len(raw)!=2*bits*size or any(v>self.mask for v in vals):
                raise ValueError('Invalid Boolean planes')
            return [vals[:bits],vals[bits:]]
        for k,personal,terminal,teacher,meta in data['rows']:
            if not isinstance(k,str) or not k.isdecimal() or k in self.rows or codec is not None and not codec.validate(k):
                raise ValueError('Invalid memory address')
            if not 0<=meta['decayed']<=self.clock or len(meta['sources'])>L.MEMORY_SOURCE_LIMIT:
                raise ValueError('Invalid memory metadata')
            self.rows[k]=unpack(personal);self.meta[k]=dict(meta)
            if terminal is not None:self.terminal[k]=unpack(terminal)
            if teacher is not None:self.teachers[k]=unpack(teacher)
            if all(v is None or (v.islower() or v.isdecimal()) and ' ' not in v for v in (personal,terminal,teacher)):
                self._packed[k]=(personal,terminal,teacher)  # canonical hex: identical to a fresh export
