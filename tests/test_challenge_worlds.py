import json
from pathlib import Path
import random
import tempfile
import unittest

from boolean_engine import VectorBooleanNet
from engine.challenge_worlds import make_world
from engine.challenges import (build_scenario, create_maze_map, load_suites, prepare_suites,
                               scenario_names, seeds_for, training_generation)
from engine.evaluation import baseline_suite, rollout, sensor_greedy
from engine.world import DIRS, World


class ChallengeWorldTests(unittest.TestCase):
    def test_classic_suite_and_dynamic_seeds_preserved(self):
        names = ('static','moving','maze','hidden')
        self.assertEqual(scenario_names('classic'), names)
        self.assertEqual(seeds_for(3,'validation','hidden',2),[323000,323001])
        original = training_generation(8,91,1,2,[],'r')
        extended = training_generation(8,91,1,2,[],'r','extended')
        self.assertEqual(original[0], extended[0][:4])
        classic = original[0][0]
        a = World(classic['map'],11,classic['visibility'],classic['prey_moves'])
        b = make_world(classic,11)
        rng = random.Random(89)
        for _ in range(40):
            action=rng.randrange(8)
            self.assertEqual(a.step(action),b.step(action))
            self.assertEqual(a.observe(),b.observe())

    def test_saved_extended_maps_and_split_ids(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root,'config.json').write_text(json.dumps({'scenario_set':'extended'}))
            suites=prepare_suites(root,8,13,'extended')
            for split, scenes in suites.items():
                self.assertEqual(load_suites(root,split), scenes)
            for task in scenario_names('extended'):
                maps = [next(s for s in group if s['name']==task)['map']
                        for group in suites.values()]
                self.assertEqual(len({m['seed'] for m in maps}),3)
                self.assertEqual(len({str(m['obstacles']) for m in maps}),3)

    def test_maze_is_a_tree_with_dead_ends_and_winding_paths(self):
        for size in (6,8,9,15):
            for seed in (11,12,13):
                m = create_maze_map(size,seed)
                world = make_world({**build_scenario('maze_novel',size,seed),'map':m},seed)
                cells = set(world._free_cells)
                degrees = [sum((x+dx,y+dy) in cells for dx,dy in DIRS) for x,y in cells]
                seen={min(cells)}
                queue=list(seen)
                for x,y in queue:
                    for dx,dy in DIRS:
                        pos=x+dx,y+dy
                        if pos in cells and pos not in seen:
                            queue.append(pos);seen.add(pos)
                self.assertEqual(seen,cells)
                self.assertEqual(sum(degrees),2*(len(cells)-1))
                self.assertGreaterEqual(degrees.count(1),2)
                self.assertFalse(world.pounce_enabled)
                self.assertEqual(m,create_maze_map(size,seed))

    def test_tag_feedback_is_delayed_including_terminal_flag(self):
        scene = build_scenario('delayed_reward',8,19)
        scene['map']['obstacles'] = []
        for delay in (0,1,4):
            w=make_world({**scene,'reward_delay':delay},55)
            w.predator,w.prey=(1,1),(2,1)
            rewards=[w.step(3)]
            for _ in range(delay):
                rewards.append(w.step(1))
            self.assertEqual(rewards,[(0,False)]*delay+[(20,True)])
            self.assertEqual(w.world_events(),{'tag_tick':1,'due_tick':1+delay})
            w.reset()
            self.assertIsNone(w.tag_tick)
        with self.assertRaises(ValueError):
            make_world({**scene,'reward_delay':-1},55)

    def test_evader_net_is_independent_and_controls_movement(self):
        scene=build_scenario('intelligent_prey',8,19)
        scene['map']['obstacles']=[]
        scene['visibility']=2
        original=VectorBooleanNet(5,neurons=32,shape=(8,)).export()
        scene['prey_genome']=original
        w=make_world(scene,7)
        other=make_world(scene,7)
        self.assertIsNot(w.prey_net, other.prey_net)
        w.predator,w.prey=(0,0),(4,4)
        inputs=[]
        w.prey_net.decide=lambda obs: inputs.append(obs) or 3
        w.step(0)
        self.assertEqual(w.prey,(5,4))
        self.assertEqual(inputs[0][:5],(0,0,0,0,0))
        self.assertEqual(w.world_events()['prey_action'],3)
        self.assertEqual(other.prey_net.export(), original)
        a=VectorBooleanNet(1,neurons=32,shape=(8,))
        data=a.export()
        traces=[rollout(VectorBooleanNet.restore(data),scene,19,10,record=True)[1] for _ in range(2)]
        self.assertEqual(traces[0],traces[1])
        self.assertIn('world_events',traces[0][1])

    def test_fair_baseline_uses_sensor_contract_and_rejects_unknown_names(self):
        obs=(1,0,0,0,1,1,1,1,0,1,1,1,1,0,0)
        self.assertEqual(sensor_greedy(obs,random.Random(3)),7)
        scene=build_scenario('intelligent_prey',6,199)
        for k in ('random','sensor_greedy','greedy'):
            out=baseline_suite(k,[scene],89,'holdout',2,8)
            row=out['by_scenario']['intelligent_prey']
            self.assertEqual(row['episode_count'],2)
            self.assertIn('capture_wilson95',row)
        with self.assertRaises(ValueError):
            baseline_suite('misspelled',[scene],89,'holdout',2,8)


if __name__ == '__main__':
    unittest.main()
