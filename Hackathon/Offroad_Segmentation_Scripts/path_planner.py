
import sys
import time
import heapq
import logging
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import interpolate

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# =================================================================================================
# 1. CONFIGURATION & TYPES
# =================================================================================================

@dataclass
class PlannerConfig:
    """Configuration parameters for the Path Planner."""
    
    # Semantic Class IDs
    CLASS_LANDSCAPE: int = 7100
    CLASS_GRASS: int = 200
    CLASS_ROUGH_A: int = 300
    CLASS_ROUGH_B: int = 550
    CLASS_ROCKS: int = 800
    CLASS_OBSTACLE_SET: frozenset = frozenset([800, 10000, 0, 500])

    # Dynamic Cost Ranges (Min, Max)
    # INCREASED to force deviation around rough patches
    COST_SOFT_RANGE: Tuple[float, float] = (4.0, 15.0) 
    COST_ROUGH_RANGE: Tuple[float, float] = (8.0, 30.0)
    
    # Defaults
    COST_LANDSCAPE: float = 1.0     
    COST_SOFT_TERRAIN: float = 5.0    # Higher cost for grass
    COST_ROUGH_TERRAIN: float = 10.0 # Much higher cost for rocks/rough
    COST_OBSTACLE: float = float('inf')
    
    # Safety
    ROBOT_RADIUS_PX: int = 6       
    SAFETY_MARGIN_FACTOR: float = 1.3 
    
    # A* Settings
    # LOWER heuristic to encourage exploration (Dijkstra-like behavior)
    HEURISTIC_WEIGHT: float = 1.05 
    MAX_ITERATIONS: int = 1000000 
    
    # History Guidance - REDUCED to prevent locking into previous straight lines
    HISTORY_WEIGHT: float = 0.5     
    
    # Desperation Mode
    DESPERATION_ITERATIONS: int = 5000000

    # Helper
    TRAVERSABLE_CLASSES = [CLASS_LANDSCAPE, CLASS_GRASS, CLASS_ROUGH_A, CLASS_ROUGH_B]

# =================================================================================================
# 2. CORE PLANNER CLASS
# =================================================================================================

class OffRoadPathPlanner:
    """
    Sequence-Aware Planner with Failure Recovery.
    """
    
    def __init__(self, config: PlannerConfig = PlannerConfig()):
        self.cfg = config

    def plan_sequence(self, 
                      segmentation_mask: np.ndarray, 
                      start: Tuple[int, int], 
                      goal: Tuple[int, int],
                      prev_path_endpoint: Optional[Tuple[int, int]] = None) -> Tuple[List[Tuple[int, int]], Dict]:
        """
        Main entry point for sequence planning. Guaranteed to return a path.
        """
        t0 = time.time()
        
        # 1. Analyze Terrain Complexity
        complexity_score = self._analyze_terrain_complexity(segmentation_mask)
        logger.info(f"Terrain Complexity: {complexity_score:.2f}")
        
        # 2. Dynamic Tuning
        current_cfg = self._tune_config(complexity_score)
        
        # 3. Try Standard Plan
        path, metadata = self._attempt_plan(segmentation_mask, start, goal, current_cfg, prev_path_endpoint)
        
        if path:
            metadata['mode'] = 'standard'
            return path, metadata
            
        # 4. Fallback 1: Reduced Safety Margins (But KEEP Repulsion)
        logger.warning("Standard Plan Failed. Reducing Safety Margins...")
        current_cfg.ROBOT_RADIUS_PX = max(1, int(current_cfg.ROBOT_RADIUS_PX * 0.5))
        # Slightly greedier but still allows curvature
        current_cfg.HEURISTIC_WEIGHT = 2.0 
        path, metadata = self._attempt_plan(segmentation_mask, start, goal, current_cfg, prev_path_endpoint)
        
        if path:
            metadata['mode'] = 'reduced_safety'
            return path, metadata
            
        # 5. Fallback 2: Tunneled Costs (Allow traversal but penalize heavily)
        logger.warning("Reduced Safety Failed. Enabling Obstacle Traversal...")
        
        # Keep costs relative but make search faster
        current_cfg.COST_OBSTACLE = 100.0 
        current_cfg.COST_SOFT_TERRAIN = 5.0
        current_cfg.COST_ROUGH_TERRAIN = 10.0
        
        # LOW heuristic to allow lateral exploration and respect repulsion field
        # The strong repulsion (k=300) will dominate and force curvature
        current_cfg.HEURISTIC_WEIGHT = 1.5 
        current_cfg.MAX_ITERATIONS = 2000000 
        
        path, metadata = self._attempt_plan(segmentation_mask, start, goal, current_cfg, prev_path_endpoint)
        
        if path:
            metadata['mode'] = 'reduced_costs'
            return path, metadata
            
        # 6. Desperation Mode: Geometric Path (Ignore obstacles, just minimal collision)
        logger.error("ALL PLANS FAILED. DESPERATION MODE ENGAGED.")
        path = self._generate_straight_path(start, goal)
        metadata = {'error': 'DESPERATION', 'mode': 'desperation'}
        return path, metadata

    def _attempt_plan(self, mask, start, goal, config, prev_endpoint):
        cost_map, obs_mask = self._generate_cost_map(mask, config)
        
        if not self._is_valid_point(start, obs_mask):
             start = self._find_nearest_valid(start, obs_mask)
             if not start: return None, {}

        if not self._is_valid_point(goal, obs_mask):
             goal = self._find_nearest_valid(goal, obs_mask)
             if not goal: return None, {}
             
        path = self._astar_search(cost_map, start, goal, config, prev_endpoint)
        if path:
             # Only smooth if stricly safe or reduced safety
             if config.COST_OBSTACLE == float('inf'):
                 path = self._smooth_path(path, obs_mask, config)
             return path, {'cost_map': cost_map}
        return None, {'cost_map': cost_map}

    def _analyze_terrain_complexity(self, mask: np.ndarray) -> float:
        total_px = mask.size
        obs_px = np.sum(np.isin(mask, list(self.cfg.CLASS_OBSTACLE_SET)))
        rough_px = np.sum(np.isin(mask, [self.cfg.CLASS_ROUGH_A, self.cfg.CLASS_ROUGH_B, self.cfg.CLASS_GRASS]))
        score = (obs_px * 1.0 + rough_px * 0.5) / total_px
        return min(1.0, score)

    def _tune_config(self, complexity: float) -> PlannerConfig:
        new_cfg = PlannerConfig()
        factor = 1.0 - complexity 
        s_min, s_max = self.cfg.COST_SOFT_RANGE
        r_min, r_max = self.cfg.COST_ROUGH_RANGE
        new_cfg.COST_SOFT_TERRAIN = s_min + (s_max - s_min) * factor
        new_cfg.COST_ROUGH_TERRAIN = r_min + (r_max - r_min) * factor
        return new_cfg

    def _generate_cost_map(self, seg_mask: np.ndarray, config: PlannerConfig) -> Tuple[np.ndarray, np.ndarray]:
        h, w = seg_mask.shape
        cost_grid = np.full((h, w), config.COST_LANDSCAPE, dtype=np.float32)
        obstacle_mask = np.zeros((h, w), dtype=np.uint8)
        
        mask_grass = (seg_mask == config.CLASS_GRASS)
        cost_grid[mask_grass] = config.COST_SOFT_TERRAIN
        
        mask_rough = (seg_mask == config.CLASS_ROUGH_A) | (seg_mask == config.CLASS_ROUGH_B)
        cost_grid[mask_rough] = config.COST_ROUGH_TERRAIN
        
        is_safe = (seg_mask == config.CLASS_LANDSCAPE) | \
                  (seg_mask == config.CLASS_GRASS) | \
                  (seg_mask == config.CLASS_ROUGH_A) | \
                  (seg_mask == config.CLASS_ROUGH_B)
        
        mask_obs = ~is_safe
        obstacle_mask[mask_obs] = 1
        
        if config.COST_OBSTACLE < float('inf'):
             cost_grid[mask_obs] = config.COST_OBSTACLE
        else:
             cost_grid[mask_obs] = float('inf')

        # 3. REPULSION FIELD (The "Anti-Bush" Logic)
        # Even if obstacles are technically traversable (in fallback), 
        # we want a smoothly decaying High Cost around them to force curves.
        
        dist_input = 1 - obstacle_mask
        # Calculate distance to nearest 0 (obstacle)
        # We invert logic: distance to NEAREST OBSTACLE
        obs_points = (obstacle_mask == 1).astype(np.uint8)
        
        # If there are obstacles, compute distance FROM them
        if np.any(obs_points):
            # inverted: 0 at obst, increasing away
            dist_to_obs = cv2.distanceTransform((1-obs_points), cv2.DIST_L2, 5)
            
            # Repulsion Decay:
            # VERY High cost near obstacles, decaying over wide area
            # This MUST dominate to force curvature
            k_repulsion = 300.0 # MUCH STRONGER amplitude
            decay_rate = 40.0   # WIDER field (pixels)
            
            repulsion_field = k_repulsion * np.exp(-dist_to_obs / decay_rate)
            
            # Add to grid
            cost_grid += repulsion_field

        # 4. Inflation (Binary Check)
        if config.ROBOT_RADIUS_PX > 0:
            inflation_px = int(config.ROBOT_RADIUS_PX * config.SAFETY_MARGIN_FACTOR)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inflation_px*2 + 1, inflation_px*2 + 1))
            inflated_obstacles = cv2.dilate(obstacle_mask, kernel)
            
            # If standard mode, these are pure walls
            if config.COST_OBSTACLE == float('inf'):
                cost_grid[inflated_obstacles == 1] = float('inf')
            else:
                # If traversable, allow them but trust the Cost Map to penalize
                # We need a mask for boolean checks though
                inflated_obstacles = np.zeros_like(obstacle_mask) 
        else:
            inflated_obstacles = obstacle_mask if config.COST_OBSTACLE == float('inf') else np.zeros_like(obstacle_mask)

        return cost_grid, inflated_obstacles

    def _astar_search(self, cost_map, start, goal, config, prev_endpoint):
        h, w = cost_map.shape
        open_set = []
        heapq.heappush(open_set, (0.0, 0, start[0], start[1]))
        
        came_from = {}
        g_score = {start: 0.0}
        
        step_counter = 0
        neighbors = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        costs_move = [1.0]*4 + [1.414]*4
        
        goal_x, goal_y = goal
        
        while open_set:
            if step_counter > config.MAX_ITERATIONS:
                return None
            
            current_f, _, cx, cy = heapq.heappop(open_set)
            
            if abs(cx - goal_x) <= 2 and abs(cy - goal_y) <= 2:
                return self._reconstruct_path(came_from, (cx, cy))
            
            current_g = g_score.get((cx, cy), float('inf'))
            
            for i, (dx, dy) in enumerate(neighbors):
                nx, ny = cx + dx, cy + dy
                
                if 0 <= nx < w and 0 <= ny < h:
                    cell_cost = cost_map[ny, nx]
                    if cell_cost == float('inf'): continue
                        
                    tentative_g = current_g + (costs_move[i] * cell_cost)
                    
                    if tentative_g < g_score.get((nx, ny), float('inf')):
                        came_from[(nx, ny)] = (cx, cy)
                        g_score[(nx, ny)] = tentative_g
                        
                        h_dist = np.hypot(goal_x - nx, goal_y - ny)
                        h_history = 0.0
                        if prev_endpoint:
                             px, py = prev_endpoint
                             h_history = np.hypot(px - nx, py - ny) * config.HISTORY_WEIGHT
                        
                        f_score = tentative_g + (h_dist * config.HEURISTIC_WEIGHT) + h_history
                        
                        step_counter += 1
                        heapq.heappush(open_set, (f_score, step_counter, nx, ny))
        return None

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def _smooth_path(self, path, obstacle_mask, config):
        if len(path) < 5: return path
        if config.COST_OBSTACLE < float('inf'): return path 
        
        try:
             path_arr = np.array(path)
             x, y = path_arr[:, 0], path_arr[:, 1]
             if len(path) > 20:
                 indices = np.arange(0, len(path), 5)
                 if indices[-1] != len(path)-1:
                     indices = np.append(indices, len(path)-1)
                 x = x[indices]
                 y = y[indices]
             
             tck, u = interpolate.splprep([x, y], s=config.SMOOTHING_FACTOR, k=3)
             u_new = np.linspace(0, 1, num=len(path)) 
             x_new, y_new = interpolate.splev(u_new, tck)
             
             smoothed = []
             h, w = obstacle_mask.shape
             for xi, yi in zip(x_new, y_new):
                 r_x, r_y = int(round(xi)), int(round(yi))
                 if 0 <= r_x < w and 0 <= r_y < h:
                     if obstacle_mask[r_y, r_x] == 1:
                         return path 
                     smoothed.append((r_x, r_y))
             return smoothed
        except:
             return path

    def _is_valid_point(self, pt, mask):
        x, y = pt
        h, w = mask.shape
        if not (0 <= x < w and 0 <= y < h): return False
        if mask[y, x] == 1: return False
        return True

    def _find_nearest_valid(self, pt, mask, max_radius=50):
        x, y = pt
        h, w = mask.shape
        for r in range(1, max_radius):
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if mask[ny, nx] == 0:
                            return (nx, ny)
        return None

    def _generate_straight_path(self, start, goal):
        x0, y0 = start
        x1, y1 = goal
        num = int(np.hypot(x1-x0, y1-y0))
        x = np.linspace(x0, x1, num)
        y = np.linspace(y0, y1, num)
        return list(zip(x.astype(int), y.astype(int)))
