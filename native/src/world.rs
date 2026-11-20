//! Grid worlds and the episode loop (port of engine/world.py, challenge_worlds.py, evaluation.rollout).
use crate::circuit::Brain;
use std::collections::HashSet;
use tesseract_core::Rng;

pub const DIRS: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];

#[derive(Clone, Debug)]
pub struct Geometry {
    pub size: i32,
    pub blocked: Vec<bool>,
    pub free_cells: Vec<(i32, i32)>,
    pub obstacle_count: usize,
}

impl Geometry {
    pub fn new(size: i32, obstacles: &[(i32, i32)]) -> Geometry {
        let mut blocked = vec![false; (size * size) as usize];
        let mut unique = HashSet::new();
        for &(x, y) in obstacles {
            unique.insert((x, y));
            if x >= 0 && y >= 0 && x < size && y < size {
                blocked[(y * size + x) as usize] = true;
            }
        }
        let mut free_cells = Vec::new();
        for y in 0..size {
            for x in 0..size {
                if !blocked[(y * size + x) as usize] {
                    free_cells.push((x, y));
                }
            }
        }
        Geometry { size, blocked, free_cells, obstacle_count: unique.len() }
    }
    #[inline]
    pub fn is_obstacle(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && x < self.size && y < self.size && self.blocked[(y * self.size + x) as usize]
    }
    #[inline]
    pub fn free(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && x < self.size && y < self.size && !self.blocked[(y * self.size + x) as usize]
    }
    fn line_of_sight(&self, a: (i32, i32), b: (i32, i32)) -> bool {
        let (mut x0, mut y0) = a;
        let (x1, y1) = b;
        let dx = (x1 - x0).abs();
        let dy = (y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx - dy;
        while (x0, y0) != (x1, y1) {
            let e2 = 2 * err;
            if e2 > -dy { err -= dy; x0 += sx; }
            if e2 < dx { err += dx; y0 += sy; }
            if self.is_obstacle(x0, y0) { return false; }
        }
        true
    }
    /// 15 sensor bits packed little-endian (bit i = sensor i).
    pub fn observe_from(&self, observer: (i32, i32), target: (i32, i32), collisions: u64, visibility: i64) -> u64 {
        let (x, y) = observer;
        let (px, py) = target;
        let distance = ((x - px).abs() + (y - py).abs()) as i64;
        let visible = distance <= visibility && self.line_of_sight(observer, target);
        let mut bitsv: Vec<bool> = vec![visible, visible && py < y, visible && py > y, visible && px < x, visible && px > x];
        for d in [1, 2] {
            for (dx, dy) in DIRS {
                bitsv.push(self.free(x + dx * d, y + dy * d));
            }
        }
        bitsv.push(visible && distance <= 2);
        bitsv.push(collisions > 0);
        bitsv.iter().enumerate().fold(0u64, |acc, (i, &b)| acc | ((b as u64) << i))
    }
}

pub enum Kind {
    Plain { pounce: bool },
    Delayed { delay: u64 },
    Neural,
}

pub struct EpisodeResult {
    pub capture: bool,
    pub steps: u64,
    pub reward: f64,
    pub stall: bool,
    pub blocked: u64,
    pub exploration: f64,
    pub final_distance: i64,
    pub obstacles: usize,
    pub predator: (i32, i32),
    pub prey: (i32, i32),
}

struct World<'g> {
    g: &'g Geometry,
    rng: Rng,
    visibility: i64,
    prey_moves: bool,
    pounce: bool,
    predator: (i32, i32),
    prey: (i32, i32),
    steps: u64,
    collisions: u64,
    captured: bool,
    visited: HashSet<(i32, i32)>,
    // delayed reward
    delay: Option<u64>,
    tag_tick: Option<u64>,
    due_tick: Option<u64>,
    // neural prey
    neural: bool,
    prey_collisions: u64,
    pending_reward: Option<f64>,
}

impl<'g> World<'g> {
    fn reset(&mut self) {
        let n = self.g.free_cells.len() as u64;
        let i = self.rng.below(n);
        let mut j = self.rng.below(n - 1);
        if j >= i { j += 1; }
        self.predator = self.g.free_cells[i as usize];
        self.prey = self.g.free_cells[j as usize];
        self.steps = 0;
        self.collisions = 0;
        self.captured = false;
        self.visited.clear();
        self.visited.insert(self.predator);
    }

    fn observe(&self) -> u64 {
        self.g.observe_from(self.predator, self.prey, self.collisions, self.visibility)
    }

    fn move_prey_random(&mut self) {
        let mut options = vec![self.prey];
        for (u, v) in DIRS {
            options.push((self.prey.0 + u, self.prey.1 + v));
        }
        let free: Vec<(i32, i32)> = options.into_iter().filter(|p| self.g.free(p.0, p.1)).collect();
        let k = self.rng.below(free.len() as u64) as usize;
        self.prey = free[k];
    }

    fn credit_previous(&mut self, prey: &mut Brain, observation: u64, caught: bool) -> Result<(), &'static str> {
        if let Some(r) = self.pending_reward.take() {
            prey.transition(observation, false);
            prey.feedback(if caught { -4.0 } else { r })?;
        }
        Ok(())
    }

    fn move_prey_neural(&mut self, prey: &mut Brain) -> Result<(), &'static str> {
        let before = self.g.observe_from(self.prey, self.predator, self.prey_collisions, self.visibility);
        self.credit_previous(prey, before, false)?;
        let action = prey.decide(before);
        let (dx, dy) = DIRS[action % 4];
        let (x, y) = self.prey;
        let near = (x + dx, y + dy);
        let far = (x + 2 * dx, y + 2 * dy);
        let landing = if action >= 4 { far } else { near };
        let legal = self.g.free(landing.0, landing.1) && (action < 4 || !self.g.free(near.0, near.1));
        if legal { self.prey = landing; } else { self.prey_collisions += 1; }
        let after = self.g.observe_from(self.prey, self.predator, self.prey_collisions, self.visibility);
        let caught = self.prey == self.predator;
        let r = if caught { -4.0 } else if !legal { -1.0 } else { ((before & 1 == 1) && (after & 1 == 0)) as i32 as f64 };
        self.pending_reward = Some(r);
        Ok(())
    }

    /// Base World.step; returns (reward, captured).
    fn base_step(&mut self, action: usize, prey: Option<&mut Brain>) -> Result<(f64, bool), &'static str> {
        let (dx, dy) = DIRS[action % 4];
        let leap = action >= 4;
        let k = if leap { 2 } else { 1 };
        let dest = (self.predator.0 + dx * k, self.predator.1 + dy * k);
        let mid = (self.predator.0 + dx, self.predator.1 + dy);
        let legal = self.g.free(dest.0, dest.1) && (!leap || (self.pounce && !self.g.free(mid.0, mid.1)));
        if legal { self.predator = dest; } else { self.collisions += 1; }
        let new_cell = self.visited.insert(self.predator);
        self.steps += 1;
        self.captured = self.predator == self.prey;
        if !self.captured && self.prey_moves {
            if self.neural {
                self.move_prey_neural(prey.expect("neural prey brain"))?;
            } else {
                self.move_prey_random();
            }
            self.captured = self.predator == self.prey;
        }
        let reward = if self.captured {
            20.0
        } else if !legal {
            -1.0
        } else {
            (if new_cell { 0.12 } else { -0.04 }) - (if leap { 0.12 } else { 0.0 })
        };
        Ok((reward, self.captured))
    }

    fn step(&mut self, action: usize, mut prey: Option<&mut Brain>) -> Result<(f64, bool), &'static str> {
        if let Some(delay) = self.delay {
            let (_, contact) = self.base_step(action, None)?;
            if contact && self.tag_tick.is_none() {
                self.tag_tick = Some(self.steps);
                self.due_tick = Some(self.steps + delay);
            }
            self.captured = matches!(self.due_tick, Some(d) if self.steps >= d);
            return Ok((if self.captured { 20.0 } else { 0.0 }, self.captured));
        }
        if self.neural {
            let (reward, caught) = self.base_step(action, prey.as_deref_mut())?;
            if caught {
                let obs = self.g.observe_from(self.prey, self.predator, self.prey_collisions, self.visibility);
                self.credit_previous(prey.unwrap(), obs, true)?;
            }
            return Ok((reward, caught));
        }
        self.base_step(action, None)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_episode(pred: &mut Brain, mut prey: Option<&mut Brain>, g: &Geometry, kind: &Kind, seed: u64,
                   visibility: i64, prey_moves: bool, steps: u64) -> Result<EpisodeResult, &'static str> {
    let mut w = World {
        g, rng: Rng::new(seed), visibility, prey_moves, pounce: true,
        predator: (0, 0), prey: (0, 0), steps: 0, collisions: 0, captured: false, visited: HashSet::new(),
        delay: None, tag_tick: None, due_tick: None, neural: false, prey_collisions: 0, pending_reward: None,
    };
    match kind {
        Kind::Plain { pounce } => w.pounce = *pounce,
        Kind::Delayed { delay } => { w.delay = Some(*delay); w.prey_moves = false; }
        Kind::Neural => {
            w.neural = true;
            w.prey_moves = true;
            prey.as_deref_mut().ok_or("neural prey brain missing")?.reset();
        }
    }
    if g.free_cells.len() < 2 {
        return Err("map needs two free cells");
    }
    w.reset();
    pred.reset();
    let mut total = 0.0f64;
    let mut blocked = 0u64;
    let mut stalled = false;
    let mut trail: Vec<(i32, i32)> = Vec::new();
    for _ in 0..steps {
        let observation = w.observe();
        let action = pred.decide(observation);
        let previous = w.predator;
        let (reward, captured) = w.step(action, prey.as_deref_mut())?;
        total += reward;
        if w.predator == previous { blocked += 1; }
        trail.push(w.predator);
        pred.transition(w.observe(), captured);
        pred.feedback(reward)?;
        if captured { break; }
        if trail.len() >= 20 && w.due_tick.is_none() {
            let last: HashSet<(i32, i32)> = trail[trail.len() - 20..].iter().copied().collect();
            if last.len() <= 2 {
                stalled = true;
                total -= 2.0;
                break;
            }
        }
    }
    Ok(EpisodeResult {
        capture: w.captured,
        steps: w.steps,
        reward: total,
        stall: stalled,
        blocked,
        exploration: w.visited.len() as f64 / g.free_cells.len() as f64,
        final_distance: ((w.predator.0 - w.prey.0).abs() + (w.predator.1 - w.prey.1).abs()) as i64,
        obstacles: g.obstacle_count,
        predator: w.predator,
        prey: w.prey,
    })
}
