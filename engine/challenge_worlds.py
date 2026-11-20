"""Optional P5 worlds. The reference curriculum still constructs the V6.05 World."""
from boolean_engine import VectorBooleanNet
from .world import DIRS, World


class DelayedRewardWorld(World):
    """Contact tags the prey. Confirmation and reward arrive after D more decisions.

    Unconfirmed tags expire at the episode budget. The learner receives no shaped
    action reward or terminal-success flag until the confirmation tick.
    """
    def __init__(self, record, seed, visibility, delay):
        if type(delay) is not int or not 0 <= delay <= 64:
            raise ValueError('reward_delay must be an integer between 0 and 64')
        self.delay = delay
        super().__init__(record, seed, visibility, prey_moves=False)

    def reset(self):
        self.tag_tick = self.due_tick = None
        return super().reset()

    def step(self, action):
        _, contact = super().step(action)
        if contact and self.tag_tick is None:
            self.tag_tick = self.steps
            self.due_tick = self.steps + self.delay
        self.captured = self.due_tick is not None and self.steps >= self.due_tick
        return (20 if self.captured else 0), self.captured

    def world_events(self):
        return {'tag_tick': self.tag_tick, 'due_tick': self.due_tick}


class NeuralPreyWorld(World):
    """A second boolean network acts from its own local sensors and learns locally."""
    def __init__(self, record, seed, visibility, brain=None, prey_seed=641):
        self.prey_net = (VectorBooleanNet.restore(brain) if brain is not None else
                         VectorBooleanNet(prey_seed, neurons=32, shape=(8,)))
        if self.prey_net.sensors != 15 or self.prey_net.action_bits != 3:
            raise ValueError('prey genome must support 15 sensors and 8 actions')
        super().__init__(record, seed, visibility, prey_moves=True)

    def reset(self):
        self.prey_net.reset()
        self.prey_collisions = 0
        self.prey_action = None
        self.pending_reward = None
        return super().reset()

    def move_prey(self):
        before = self.observe_from(self.prey, self.predator, self.prey_collisions)
        self._credit_previous(before)
        action = self.prey_net.decide(before)
        self.prey_action = action
        dx, dy = DIRS[action % 4]
        x, y = self.prey
        near = x + dx, y + dy
        far = x + 2*dx, y + 2*dy
        landing = far if action >= 4 else near
        legal = self.free(landing) and (action < 4 or not self.free(near))
        if legal:
            self.prey = landing
        else:
            self.prey_collisions += 1
        after = self.observe_from(self.prey, self.predator, self.prey_collisions)
        caught = self.prey == self.predator
        # Signed feedback expresses the evader's objective. Losing sight after
        # a legal move is an observable proxy for escape, never a privileged distance.
        self.pending_reward = -4 if caught else (-1 if not legal else int(before[0] and not after[0]))

    def _credit_previous(self, observation, caught=False):
        if self.pending_reward is not None:
            self.prey_net.transition(observation, False)
            self.prey_net.feedback(-4 if caught else self.pending_reward)
            self.pending_reward = None

    def step(self, action):
        self.prey_action = None
        reward, caught = super().step(action)
        if caught:
            self._credit_previous(self.observe_from(self.prey, self.predator, self.prey_collisions),
                                  caught=True)
        return reward, caught

    def world_events(self):
        return {'prey_action': self.prey_action,
                'prey_decoder': self.prey_net.last_decoder if self.prey_action is not None else None,
                'prey_collisions': self.prey_collisions}


def make_world(scenario, seed):
    name = scenario.get('name', scenario.get('scenario', 'static'))
    record, visibility = scenario['map'], scenario['visibility']
    if name == 'delayed_reward':
        return DelayedRewardWorld(record, seed, visibility, scenario.get('reward_delay', 4))
    if name == 'intelligent_prey':
        return NeuralPreyWorld(record, seed, visibility, scenario.get('prey_genome'),
                               scenario.get('prey_seed', 641))
    world = World(record, seed, visibility, scenario['prey_moves'])
    if name == 'maze_novel':
        world.pounce_enabled = False
    return world
