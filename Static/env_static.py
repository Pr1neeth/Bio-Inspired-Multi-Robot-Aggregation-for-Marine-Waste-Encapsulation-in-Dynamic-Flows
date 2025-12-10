from __future__ import annotations
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Literal, List, Union

import numpy as np

# ---------------------------------------------------------------------
# Optional Gymnasium vs classic Gym
# ---------------------------------------------------------------------
try:
    import gymnasium as gym
    from gymnasium import spaces
    _USING_GYMNASIUM = True
except Exception:
    import gym  # type: ignore
    from gym import spaces  # type: ignore
    _USING_GYMNASIUM = False
    print("[OceanTrashEnv] gymnasium not found; falling back to classic gym.", file=sys.stderr)

# Optional pygame for rendering
try:
    import pygame
    _HAS_PYGAME = True
except Exception:
    _HAS_PYGAME = False


# ---------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------
def _shape_tuple(shape: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)


def rng_random(rng, shape: Union[int, Tuple[int, ...]]):
    shape = _shape_tuple(shape)
    if hasattr(rng, "random"):
        return rng.random(shape)
    return rng.rand(*shape)


def rng_standard_normal(rng, shape: Union[int, Tuple[int, ...]]):
    shape = _shape_tuple(shape)
    if hasattr(rng, "standard_normal"):
        return rng.standard_normal(shape)
    return rng.randn(*shape)


def rng_integers(rng, low: int, high: int, size: Optional[Union[int, Tuple[int, ...]]] = None):
    if hasattr(rng, "integers"):
        return rng.integers(low, high, size=size)
    return rng.randint(low, high, size=size)


PlacementMode = Literal["center", "uniform"]


# ---------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------
@dataclass
class OceanTrashConfig:
    grid_size: Tuple[int, int] = (20, 30)  # (H, W)
    n_trash: int = 1
    n_robots: int = 6
    max_steps: int = 600
    spawn_prob: float = 0.0
    placement: PlacementMode = "center"
    wrap: bool = False
    # Rendering
    cell_px: int = 24
    grid_lines: bool = True
    fps: int = 10
    # Colors
    color_bg: Tuple[int, int, int] = (12, 44, 84)
    color_grid: Tuple[int, int, int] = (24, 86, 158)
    color_trash: Tuple[int, int, int] = (230, 236, 237)


@dataclass
class OceanPhysicsConfig:
    enabled: bool = False           # keep off for PPO training
    advect_trash: bool = False
    advect_robots: bool = False
    n_vortices: int = 0
    vortex_strength: float = 0.0
    core_radius: float = 1.5
    uniform_drift: float = 0.0
    uniform_omega: float = 0.0
    dt: float = 0.0
    diffusion: float = 0.0
    show_overlay: bool = False
    overlay_stride: int = 3


# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------
class OceanTrashEnv(gym.Env):
    """
    2D grid env:
      - One trash cell in the center (by default).
      - N robots start near top-left.
      - Actions: MultiDiscrete(5): 0=stay, 1=up, 2=down, 3=left, 4=right.
      - Observations: Dict(robot_positions, trash_map, steps).
      - Goal: robots form a circle (ring) around trash.
      - Episode ends when ring is formed (not when trash is picked up).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self,
                 grid_size: Tuple[int, int] = (24, 36),
                 n_trash: int = 1,
                 n_robots: int = 6,
                 max_steps: int = 600,
                 render_mode: Optional[str] = None,
                 spawn_prob: float = 0.0,
                 placement: PlacementMode = "center",
                 wrap: bool = False,
                 cell_px: int = 24,
                 grid_lines: bool = True,
                 fps: int = 10,
                 reward_step: float = -0.01,
                 reward_collision: float = -0.1,
                 reward_circle_scale: float = -0.1,
                 reward_circle_bonus: float = 20.0,
                 circle_radius: int = 4,
                 frozen: bool = False,
                 physics: Optional[OceanPhysicsConfig] = None,
                 seed: Optional[int] = None):
        super().__init__()

        H, W = grid_size
        self.cfg = OceanTrashConfig(
            grid_size=grid_size,
            n_trash=n_trash,
            n_robots=n_robots,
            max_steps=max_steps,
            spawn_prob=spawn_prob,
            placement=placement,
            wrap=wrap,
            cell_px=cell_px,
            grid_lines=grid_lines,
            fps=fps,
        )

        assert H >= 4 and W >= 4, "grid_size must be at least 4x4"
        assert self.cfg.n_trash == 1, "this version assumes exactly 1 trash cell"
        assert self.cfg.n_robots > 0, "n_robots must be >= 1"

        self.render_mode = render_mode
        if render_mode is not None:
            assert render_mode in self.metadata["render_modes"]

        # RNG
        if _USING_GYMNASIUM:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            self.np_random = np.random.RandomState(seed)

        # Rewards / behavior
        self.reward_step = float(reward_step)
        self.reward_collision = float(reward_collision)
        self.reward_circle_scale = float(reward_circle_scale)
        self.reward_circle_bonus = float(reward_circle_bonus)
        self.circle_radius = int(circle_radius)
        self.frozen = bool(frozen)

        # Physics (disabled by default, but kept for completeness)
        self.physics = physics if physics is not None else OceanPhysicsConfig()
        self._vortex_params = None

        # State
        self._trash_map: np.ndarray = np.zeros(self.cfg.grid_size, dtype=np.int8)
        self._robots: np.ndarray = np.zeros((self.cfg.n_robots, 2), dtype=np.int32)
        self._step_count: int = 0
        self._trash_particles: Optional[np.ndarray] = None

        # ---- Spaces ---------------------------------------------------
        low_rp = np.zeros((self.cfg.n_robots, 2), dtype=np.int32)
        high_rp = np.tile(np.array([H - 1, W - 1], dtype=np.int32), (self.cfg.n_robots, 1))
        self.action_space = spaces.MultiDiscrete([5] * self.cfg.n_robots)

        # IMPORTANT: steps is 1D (shape=(1,)) so SB3 can flatten it
        self.observation_space = spaces.Dict({
            "robot_positions": spaces.Box(low=low_rp, high=high_rp, dtype=np.int32),
            "trash_map": spaces.Box(low=0, high=1, shape=(H, W), dtype=np.int8),
            "steps": spaces.Box(
                low=np.array([0], dtype=np.int32),
                high=np.array([self.cfg.max_steps], dtype=np.int32),
                shape=(1,),
                dtype=np.int32,
            ),
        })

        # Rendering internals
        self._screen = None
        self._surface = None
        self._clock = None
        self._robot_colors = self._gen_robot_colors(self.cfg.n_robots)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        # Seeding
        if _USING_GYMNASIUM:
            super().reset(seed=seed)
            if seed is not None:
                self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            if seed is not None:
                self.np_random = np.random.RandomState(seed)

        H, W = self.cfg.grid_size
        self._trash_map.fill(0)
        self._robots.fill(0)
        self._step_count = 0
        self._trash_particles = None

        # --- Trash: exactly one in the center ---
        r0, c0 = H // 2, W // 2
        self._trash_map[r0, c0] = 1

        # --- Robots: start near top-left corner ---
        # simple pattern: row 0, columns 0..n_robots-1 (clipped to W)
        for i in range(self.cfg.n_robots):
            r = 0
            c = min(i, W - 1)
            self._robots[i] = (r, c)

        self._init_physics()

        obs = self._get_obs()
        info = {
            "remaining_trash": int(self._trash_map.sum()),
            "step": self._step_count,
            "circle_formed": False,
        }
        return obs, info

    def step(self, action):
        H, W = self.cfg.grid_size

        prev_positions = self._robots.copy()
        proposed = self._robots.copy()
        collisions = 0

        if not self.frozen:
            action = np.asarray(action, dtype=np.int32)
            assert action.shape == (self.cfg.n_robots,), \
                f"Expected action shape {(self.cfg.n_robots,)}, got {action.shape}"

            # Decode moves
            for i, a in enumerate(action):
                dr, dc = 0, 0
                if a == 1:   # up
                    dr = -1
                elif a == 2:  # down
                    dr = 1
                elif a == 3:  # left
                    dc = -1
                elif a == 4:  # right
                    dc = 1
                nr, nc = proposed[i, 0] + dr, proposed[i, 1] + dc

                if self.cfg.wrap:
                    nr %= H
                    nc %= W
                else:
                    nr = int(np.clip(nr, 0, H - 1))
                    nc = int(np.clip(nc, 0, W - 1))
                proposed[i] = (nr, nc)

            # Resolve collisions: multiple robots into same cell
            _, inv, counts = np.unique(proposed, axis=0, return_inverse=True, return_counts=True)
            for cell_id, cnt in enumerate(counts):
                if cnt <= 1:
                    continue
                idxs = np.where(inv == cell_id)[0]
                allow = int(self.np_random.choice(idxs))
                for idx in idxs:
                    if idx != allow:
                        proposed[idx] = prev_positions[idx]
                        collisions += 1

            # Swaps (i <-> j)
            for i in range(self.cfg.n_robots):
                for j in range(i + 1, self.cfg.n_robots):
                    if (
                        proposed[i][0] == prev_positions[j][0]
                        and proposed[i][1] == prev_positions[j][1]
                        and proposed[j][0] == prev_positions[i][0]
                        and proposed[j][1] == prev_positions[i][1]
                    ):
                        proposed[i] = prev_positions[i]
                        proposed[j] = prev_positions[j]
                        collisions += 1

        self._robots = proposed

        # Physics (off by default here)
        if self.physics.enabled:
            self._apply_physics()

        # --- NO PICKUP: trash never disappears ---
        collected = 0

        # --- Circle-shaped reward ---
        targets = self._ring_targets(radius=self.circle_radius)
        circle_reward = 0.0
        if targets is not None:
            # L1 distance between each robot and its assigned ring position
            dists = np.abs(self._robots - targets).sum(axis=1)  # shape (n_robots,)
            mean_dist = float(dists.mean())
            # scaled negative distance (closer → higher reward)
            circle_reward = self.reward_circle_scale * mean_dist

        reward = self.reward_step
        if not self.frozen:
            reward += self.reward_collision * float(collisions)
        reward += circle_reward

        # --- Termination: circle formed, not trash picked up ---
        self._step_count += 1
        remaining = int(self._trash_map.sum())  # always 1 here
        circle_formed = self._is_circle_formed(radius=self.circle_radius)

        terminated = False
        if circle_formed and not self.frozen:
            reward += self.reward_circle_bonus
            terminated = True

        truncated = (self._step_count >= self.cfg.max_steps)

        obs = self._get_obs()
        info = {
            "collisions": int(collisions),
            "remaining_trash": remaining,
            "step": self._step_count,
            "circle_formed": circle_formed,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Circle helpers
    # ------------------------------------------------------------------
    def _ring_targets(self, radius: int = 4):
        """Return ideal ring positions around the (single) trash cell."""
        H, W = self.cfg.grid_size
        ty, tx = np.where(self._trash_map == 1)
        if len(ty) != 1:
            return None
        tr, tc = int(ty[0]), int(tx[0])

        targets = []
        for i in range(self.cfg.n_robots):
            ang = 2.0 * np.pi * (i / self.cfg.n_robots)
            dr = int(round(radius * np.sin(ang)))
            dc = int(round(radius * np.cos(ang)))
            rr = int(np.clip(tr + dr, 0, H - 1))
            cc = int(np.clip(tc + dc, 0, W - 1))
            targets.append((rr, cc))
        return np.array(targets, dtype=np.int32)

    def _is_circle_formed(self, radius: int = 4) -> bool:
        targets = self._ring_targets(radius)
        if targets is None:
            return False
        robots_set = set(map(tuple, self._robots))
        targets_set = set(map(tuple, targets))
        return robots_set == targets_set

    # ------------------------------------------------------------------
    # Physics helpers (mostly no-op here)
    # ------------------------------------------------------------------
    def _init_physics(self):
        self._vortex_params = None

    def _apply_physics(self):
        # Physics disabled in this version
        return

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self):
        return {
            "robot_positions": self._robots.copy(),
            "trash_map": self._trash_map.copy(),
            "steps": np.array([self._step_count], dtype=np.int32),
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            raise RuntimeError("render() called but render_mode is None.")

        if self.render_mode == "human":
            self._render_frame()
            return None
        elif self.render_mode == "rgb_array":
            frame = self._render_frame()
            arr = pygame.surfarray.array3d(frame)
            arr = np.transpose(arr, (1, 0, 2))
            return arr

    def _render_frame(self):
        if not _HAS_PYGAME:
            raise ImportError("Pygame is required for rendering. Install with `pip install pygame`.")

        H, W = self.cfg.grid_size
        cell = self.cfg.cell_px
        window_w, window_h = W * cell, H * cell

        if self._screen is None:
            pygame.init()
            flags = pygame.SCALED
            self._screen = pygame.display.set_mode((window_w, window_h), flags)
            pygame.display.set_caption("OceanTrashEnv (circle formation goal)")
            self._surface = pygame.Surface((window_w, window_h))
            self._clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._screen = None
                return self._surface

        self._surface.fill(self.cfg.color_bg)

        if self.cfg.grid_lines:
            for r in range(H + 1):
                y = r * cell
                pygame.draw.line(self._surface, self.cfg.color_grid, (0, y), (window_w, y), 1)
            for c in range(W + 1):
                x = c * cell
                pygame.draw.line(self._surface, self.cfg.color_grid, (x, 0), (x, window_h), 1)

        # Trash
        trash_y, trash_x = np.where(self._trash_map == 1)
        for r, c in zip(trash_y, trash_x):
            sz = max(2, cell // 3)
            x0 = c * cell + (cell - sz) // 2
            y0 = r * cell + (cell - sz) // 2
            pygame.draw.rect(self._surface, self.cfg.color_trash, pygame.Rect(x0, y0, sz, sz))

        # Robots
        for i in range(self.cfg.n_robots):
            r, c = int(self._robots[i, 0]), int(self._robots[i, 1])
            cx = c * cell + cell // 2
            cy = r * cell + cell // 2
            radius = max(4, int(cell * 0.40))
            pygame.draw.circle(self._surface, self._robot_colors[i], center=(cx, cy), radius=radius)
            font = pygame.font.SysFont(None, max(12, int(cell * 0.40)))
            label = font.render(str(i), True, (0, 0, 0))
            rect = label.get_rect(center=(cx, cy))
            self._surface.blit(label, rect)

        font = pygame.font.SysFont(None, max(14, int(cell * 0.3)))
        status = f"Step {self._step_count}"
        text_surf = font.render(status, True, (255, 255, 255))
        self._surface.blit(text_surf, (8, 8))

        self._screen.blit(self._surface, (0, 0))
        pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(self.cfg.fps)

        return self._surface

    def close(self):
        if self._screen is not None:
            pygame.display.quit()
            pygame.quit()
        self._screen = None
        self._surface = None
        self._clock = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _gen_robot_colors(n: int) -> List[Tuple[int, int, int]]:
        colors = []
        for i in range(n):
            h = i / max(1, n)
            colors.append(OceanTrashEnv._hsv_to_rgb(h, 0.7, 0.95))
        return colors

    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float):
        i = int(h * 6.0) % 6
        f = h * 6.0 - int(h * 6.0)
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        return (int(r * 255), int(g * 255), int(b * 255))


# ---------------------------------------------------------------------
# Simple random-policy demo
# ---------------------------------------------------------------------
def _demo():
    print("Random-policy demo. Close window or Ctrl+C to exit.")
    env = OceanTrashEnv(
        grid_size=(24, 36),
        n_trash=1,
        n_robots=6,
        render_mode="human",
        physics=OceanPhysicsConfig(enabled=False),
    )
    obs, info = env.reset(seed=0)
    try:
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    _demo()
