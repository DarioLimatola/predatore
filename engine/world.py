"""Deterministic grid world. Agent observations never expose hidden prey coordinates."""
import json
import random
from collections import deque
from functools import lru_cache
from pathlib import Path

DIRS = ((0,-1),(0,1),(-1,0),(1,0))
ACTIONS = ('UP','DOWN','LEFT','RIGHT','POUNCE_UP','POUNCE_DOWN','POUNCE_LEFT','POUNCE_RIGHT')


def create_map(path, size=20, density=.18, seed=42):
    """Create a connected free-space map with a reproducible obstacle layout."""
    rng = random.Random(seed)
    if not (6 <= size <= 60 and 0 <= density <= .38):
        raise ValueError('size must be 6..60 and density 0..0.38')
    for _ in range(300):
        obstacles = {(x,y) for y in range(size) for x in range(size) if rng.random() < density}
        free = {(x,y) for y in range(size) for x in range(size)} - obstacles
        if not free: continue
        origin = next(iter(free)); seen={origin}; todo=deque([origin])
        while todo:
            x,y=todo.popleft()
            for dx,dy in DIRS:
                q=(x+dx,y+dy)
                if q in free and q not in seen: seen.add(q); todo.append(q)
        if len(seen)==len(free) and len(free)>=size*size*.55: break
    else:
        # Connectivity is guaranteed by starting from a blank grid and leaving it blank.
        obstacles=set()
    record={'version':1,'name':Path(path).stem,'size':size,'density':density,'seed':seed,
            'obstacles':[list(p) for p in sorted(obstacles)]}
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(record,indent=2),encoding='utf-8')
    return record


@lru_cache(maxsize=48)
def _compiled_geometry(size, obstacles):
    """Immutable, shared grid geometry; never shares actor state or randomness."""
    blocked = frozenset(obstacles)
    free_cells = tuple((x,y) for y in range(size) for x in range(size)
                       if (x,y) not in blocked)
    masks = {}
    for x,y in free_cells:
        masks[(x,y)] = tuple(0 <= x+dx*d < size and 0 <= y+dy*d < size
                             and (x+dx*d, y+dy*d) not in blocked
                             for d in (1,2) for dx,dy in DIRS)
    return blocked, free_cells, masks


class World:
    def __init__(self, record, seed=0, visibility=4, prey_moves=True):
        self.map=record; self.size=record['size']
        self.obstacles, self._free_cells, self._neighbor_flags = _compiled_geometry(
            self.size, tuple(tuple(p) for p in record['obstacles']))
        self.rng=random.Random(seed); self.visibility=visibility; self.prey_moves=prey_moves
        self.predator=(0,0); self.prey=(0,0); self.steps=0; self.collisions=0; self.captured=False
        self.visited=set();self.reset()

    def free(self,p):
        return 0<=p[0]<self.size and 0<=p[1]<self.size and p not in self.obstacles

    def reset(self):
        self.predator,self.prey=self.rng.sample(self._free_cells,2)
        self.steps=0;self.collisions=0;self.captured=False;self.visited={self.predator}
        return self.observe()

    def observe(self):
        """Only local perception; no global distance or position leak when prey is hidden."""
        return self.observe_from(self.predator, self.prey, self.collisions)

    def observe_from(self, observer, target, collisions=0):
        x,y=observer;px,py=target
        distance=abs(x-px)+abs(y-py)
        visible=distance<=self.visibility and self._line_of_sight(observer,target)
        neighbors = self._neighbor_flags.get(observer)
        if neighbors is None:
            # Preserve observation behavior for manually positioned test agents.
            neighbors = tuple(self.free((x + dx*d, y + dy*d))
                              for d in (1,2) for dx,dy in DIRS)
        # Bit order and historical cumulative-collision sensor are UNCHANGED.
        return (int(visible), int(visible and py<y), int(visible and py>y),
                int(visible and px<x), int(visible and px>x),
                *map(int, neighbors), int(visible and distance<=2),
                int(collisions>0))

    def _line_of_sight(self,a,b):
        # Grid ray tracing prevents seeing through walls, including diagonal rays.
        x0,y0=a;x1,y1=b;dx=abs(x1-x0);dy=abs(y1-y0)
        sx=1 if x0<x1 else -1;sy=1 if y0<y1 else -1;err=dx-dy
        while (x0,y0)!=(x1,y1):
            e2=2*err
            if e2>-dy:err-=dy;x0+=sx
            if e2<dx:err+=dx;y0+=sy
            if (x0,y0) in self.obstacles:return False
        return True

    def step(self,action):
        if not 0<=action<8:raise ValueError('invalid action')
        dx,dy=DIRS[action%4]; leap=action>=4
        dest=(self.predator[0]+dx*(2 if leap else 1),self.predator[1]+dy*(2 if leap else 1))
        # A pounce jumps over exactly one blocked cell; landing must be free.
        mid=(self.predator[0]+dx,self.predator[1]+dy)
        legal=self.free(dest) and (not leap or (getattr(self,'pounce_enabled',True) and not self.free(mid)))
        if legal:self.predator=dest
        else:self.collisions+=1
        new_cell=self.predator not in self.visited
        self.visited.add(self.predator)
        self.steps+=1
        self.captured=self.predator==self.prey
        if not self.captured and self.prey_moves:
            self.move_prey()
            self.captured=self.predator==self.prey
        # Sparse terminal feedback only: no privileged distance-based shaping.
        reward=20 if self.captured else (-1.0 if not legal else (.12 if new_cell else -.04) - (.12 if leap else 0))
        return reward,self.captured

    def move_prey(self):
        options=[self.prey]+[(self.prey[0]+u,self.prey[1]+v) for u,v in DIRS]
        self.prey=self.rng.choice([p for p in options if self.free(p)])

    def snapshot(self):
        return {'predator':list(self.predator),'prey':list(self.prey),'steps':self.steps,
                'collisions':self.collisions,'captured':self.captured,'rng':repr(self.rng.getstate())}
