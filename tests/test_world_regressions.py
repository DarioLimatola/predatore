"""Regression tests for observation encoding and shared grid geometry."""
import random
import tempfile
import unittest
from collections import deque
from pathlib import Path

from engine.world import World, create_map, DIRS


def reference_observation(world):
    x,y=world.predator; px,py=world.prey
    d=abs(x-px)+abs(y-py)
    visible=d<=world.visibility and world._line_of_sight(world.predator,world.prey)
    clear=[world.free((x+dx,y+dy)) for dx,dy in DIRS]
    land=[world.free((x+2*dx,y+2*dy)) for dx,dy in DIRS]
    return tuple(int(v) for v in (visible,visible and py<y,visible and py>y,
                visible and px<x,visible and px>x,*clear,*land,
                visible and d<=2,world.collisions>0))


class OptimizationRegressions(unittest.TestCase):
    def test_observation_exactly_matches_original(self):
        with tempfile.TemporaryDirectory() as d:
            for size in (8,12,20):
                for density in (0,.06,.2):
                    m=create_map(Path(d)/f'{size}-{density}.json',size,density,seed=91)
                    w=World(m,seed=41,visibility=5)
                    positions=random.Random(77).sample(list(w._free_cells),min(30,len(w._free_cells)))
                    for p in positions:
                        w.predator=p
                        for collisions in (0,1,3):
                            w.collisions=collisions
                            self.assertEqual(w.observe(),reference_observation(w))
                            self.assertEqual(len(w.observe()),15)

    def test_world_geometry_shared_and_rng_independent(self):
        m={'size':20,'obstacles':[[3,4],[5,6],[7,8]]}
        a=World(m,seed=18);b=World(m,seed=18)
        self.assertIs(a._free_cells,b._free_cells)
        self.assertIs(a._neighbor_flags,b._neighbor_flags)
        for action in (3,4,0,2,5,1,7):
            self.assertEqual(a.observe(),b.observe())
            self.assertEqual(a.step(action),b.step(action))

    def test_connected_challenge_map_contains_obstacles_and_all_free_cells_connect(self):
        from engine.challenges import create_connected_map
        with tempfile.TemporaryDirectory() as tmp:
            record=create_connected_map(Path(tmp)/'maze.json',40,.13,91)
        blocked={tuple(cell) for cell in record['obstacles']}
        free={(x,y) for x in range(40) for y in range(40)}-blocked
        self.assertGreater(len(blocked),10)
        start=min(free);seen={start};queue=deque([start])
        while queue:
            x,y=queue.popleft()
            for dx,dy in DIRS:
                position=(x+dx,y+dy)
                if position in free and position not in seen:
                    seen.add(position);queue.append(position)
        self.assertEqual(seen,free)

if __name__=='__main__':unittest.main()
