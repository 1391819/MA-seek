"""

    Seek game - MA

"""

import pygame
import random
from pygame import gfxdraw
import math
import time
from enum import Enum
import numpy as np

# Constants
WHITE = (150, 150, 150)
LUMBER = (63, 51, 81)
TURQUOISE = (64, 224, 208)
YELLOW = (253, 253, 150)
BLACK = (29, 28, 26)
TRANSPARENT_RADAR = (150, 150, 150, 80)
COLLISION_COLOUR = (255, 0, 0) # checking collisions (walls and boundaries) - eval
ORANGE = (255, 165, 0)

class Rotation(Enum):
    UP = 0 # 1
    RIGHT = 1 # 90
    DOWN = 2 # -1
    LEFT = 3 # -90

# GAME ENV SETTINGS
WINDOW_WIDTH = 880
WINDOW_HEIGHT = 880
FPS = 60
BLOCKSIZE = 80 # size of the grid block

# added here to avoid memory problems -> there are definitely better solutions to this
display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("GameAI")

# GAME INTERNAL SETTINGS
N_WALLS = 15
WALLS_BOUNDARY = 0 # distance of wall spawning from world size
SHOW_RADAR_RAYS = False

class GameAI:
    def __init__(self, w=WINDOW_WIDTH, h=WINDOW_HEIGHT):
        pygame.init()
        # pygame settings
        self.w = w
        self.h = h
        self.display = display
        self.clock = pygame.time.Clock()
        self.display.fill(LUMBER)
        self.n_hiders = 1 # number of hider agents
        self.n_seekers = 1 # number of seeker agents

        # game data
        self.hiders = []
        self.seekers = []
        self.walls = []

        # spawning game objects and agents
        self.spawn_walls() # spawning walls
        self.spawn_hiders() # spawning hiders
        self.spawn_seekers() # spawning seekers

    # spawning functions -------------------------------------------------------

    # draw the game grid
    def draw_grid(self):
        self.display.fill(LUMBER)
        for x in range(0, self.w, BLOCKSIZE):
            for y in range(0, self.h, BLOCKSIZE):
                rect = pygame.Rect(x, y, BLOCKSIZE, BLOCKSIZE)
                pygame.draw.rect(self.display, WHITE, rect, 1)

    # spawn walls
    def spawn_walls(self):
        # internal wall
        for i in range(N_WALLS):
            wall_x = random.randrange(BLOCKSIZE*WALLS_BOUNDARY, self.w - BLOCKSIZE*WALLS_BOUNDARY, BLOCKSIZE)
            wall_y = random.randrange(BLOCKSIZE*WALLS_BOUNDARY, self.h - BLOCKSIZE*WALLS_BOUNDARY, BLOCKSIZE)
            wall = pygame.Rect(wall_x, wall_y, BLOCKSIZE, BLOCKSIZE)
            self.walls.append(wall)

        # boundary wall
        boundary_wall = self.display.get_rect()
        self.walls.append(boundary_wall)

    # spawn hiders
    def spawn_hiders(self):
        for i in range(self.n_hiders):
            hider_idx = f"hider_{i}"
            hider_peep = Hider(hider_idx, self.walls, self.seekers)
            self.hiders.append(hider_peep)

    # spawn seekers
    def spawn_seekers(self):
        for i in range(self.n_seekers):
            seeker_idx = f"seeker_{i}"
            seeker_peep = Seeker(seeker_idx, self.walls, self.hiders)
            self.seekers.append(seeker_peep)

    # drawing functions --------------------------------------------------------

    # draw the walls
    def draw_walls(self):
        for wall in self.walls:
            if wall.width == BLOCKSIZE: # we draw only the internal walls, not the boundary one
                pygame.draw.rect(self.display, BLACK, wall)

    # draw seekers
    def draw_seekers(self):
        for seeker in self.seekers:
            seeker.draw(self.display, self.walls)
            seeker.clear_radar()

    # draw hiders
    def draw_hiders(self):
        for hider in self.hiders:
            hider.draw(self.display, self.walls)
            hider.clear_radar()

    # perform action - seeker
    # rotation: 0 up (1), 1 right (90), 2 down (-1), 3 left (-90)
    # movement: 0 no movement, 1 move forward
    def action_seekers(self, action=None):

        action = self.convert_action_from_discrete_to_multidiscrete(action)

        # action ---------------------------------------------
        # clockwise rotation
        clock_wise = [Rotation.UP, Rotation.RIGHT, Rotation.DOWN, Rotation.LEFT]
        idx = clock_wise.index(self.seekers[0].rotation)

        # 0 up (1), 1 right (90), 2 down (-1), 3 left (-90)
        if action[0] == 0:
            new_rot = clock_wise[idx] # no change if same rotation
        elif action[0] == 1:
            next_idx = (idx + 1) % 4
            new_rot = clock_wise[next_idx]
        elif action[0] == 2:
            next_idx = (idx + 1) % 4
            new_rot = clock_wise[next_idx]
        elif action[0] == 3:
            next_idx = (idx + 1) % 4
            new_rot = clock_wise[next_idx]

        self.seekers[0].rotation = new_rot

        # movement (0 no movement, 1 move forward)
        if action[1] == 0:
            move = 0
        elif action[1] == 1:
            move = 1

        self.seekers[0].move(self.walls, self.hiders, move)

    # perform action - hider
    # rotation: 0 up (1), 1 right (90), 2 down (-1), 3 left (-90)
    # movement: 0 no movement, 1 move forward
    def action_hiders(self, action=None):

        action = self.convert_action_from_discrete_to_multidiscrete(action)

        # action ---------------------------------------------
        # clockwise rotation
        clock_wise = [Rotation.UP, Rotation.RIGHT, Rotation.DOWN, Rotation.LEFT]
        idx = clock_wise.index(self.hiders[0].rotation)

        # 0 up (1), 1 right (90), 2 down (-1), 3 left (-90)
        if action[0] == 0:
            new_rot = clock_wise[idx] # no change if same rotation
        elif action[0] == 1:
            next_idx = (idx + 1) % 4
            new_rot = clock_wise[next_idx]
        elif action[0] == 2:
            next_idx = (idx + 1) % 4
            new_rot = clock_wise[next_idx]
        elif action[0] == 3:
            next_idx = (idx + 1) % 4
            new_rot = clock_wise[next_idx]

        self.hiders[0].rotation = new_rot

        # movement (0 no movement, 1 move forward)
        if action[1] == 0:
            move = 0
        elif action[1] == 1:
            move = 1

        self.hiders[0].move(self.walls, self.seekers, move)
        #self.hiders[0].reward += -0.1

    # used to convert the action from discrete to multidiscrete
    # this was a flaw in the initial design of the environment
    def convert_action_from_discrete_to_multidiscrete(self, action):

        # 0, 1, 2, 3, 4, 5, 6, 7 ->
        # 00, 01, 10, 11, 20, 21, 30, 31
        #print(action)
        if action == 0:
            action = [0, 0]
        elif action == 1:
            action = [0, 1]
        elif action == 2:
            action = [1, 0]
        elif action == 3:
            action = [1, 1]
        elif action == 4:
            action = [2, 0]
        elif action == 5:
            action = [2, 1]
        elif action == 6:
            action = [3, 0]
        elif action == 7:
            action = [3, 1]

        return action

    # evaluating step reward - seeker
    def evaluate_seekers(self):

        # if seeker caught hider
        if self.seekers[0] == self.hiders[0]:
            reward = 100
        else:
            reward = self.seekers[0].reward # internal reward is resetted before each move

            # if seeker stood still
            if self.seekers[0].moved == False:
                reward -= 3 # needs to be more than reward of dist travelled

        return reward

    # end of the episode, if hider has been found
    def is_done_seekers(self):

        if self.seekers[0] == self.hiders[0]:
            # set hider status to captured, to compute its "removal"
            # even if it moves after the seeker got on top of it
            self.hiders[0].captured = True
            return True

        return False

    # evaluating step reward - hider
    def evaluate_hiders(self):

        # if hider is on top of seeker
        if self.hiders[0] == self.seekers[0]:
            reward = -100
        else:
            reward = self.hiders[0].reward # internal reward is resetted before each move

            if self.hiders[0].moved == False:
                reward -= 3 # needs to be more than reward of dist

        return reward

    # end of the episode, if hider is on top of the seeker
    def is_done_hiders(self):

        if self.hiders[0] == self.seekers[0]:
            self.hiders[0].captured = True
            return True

        return False

    # observation space - seeker
    # we draw our peeps' rays (radars)
    # we find the first intersections of these rays
    # we return:
    #       distance from the agent
    #       object type (target, wall, empty)
    # for each ray
    def observe_seekers(self):

        # (37, 2), each row "sees" approx 5 degrees (radians) (rays/radar) and returns
        #   1) the first object met in that direction (0 = EMPTY, 1 = WALL, 2 = HIDER(s))
        #   2) its distance from the origin of the ray/radar
        #
        # # e.g.,  [150. 1]
        self.seekers[0].get_obs_data() # create it
        obs = self.seekers[0].agent_data # return it
        obs_np = np.array(obs, dtype=np.float32).flatten() # transform it (flatten)

        # clearing data once it has been processed, for next step data
        self.seekers[0].clear_agent_data()

        return obs_np

    # observation space - hider
    def observe_hiders(self):

        self.hiders[0].get_obs_data()
        obs = self.hiders[0].agent_data
        obs_np = np.array(obs, dtype=np.float32).flatten()

        self.hiders[0].clear_agent_data()

        return obs_np

    # analysis (simple way to return analysis data using the info variable)
    # we have only one so no need to use a dict for now
    def get_n_collisions_seekers(self):
        # get number of collisions every step (1 or 0)
        n_collisions = self.seekers[0].n_collisions
        # reset collision counter - could be moved inside internal class
        self.seekers[0].n_collisions = 0
        return n_collisions

    def get_n_collisions_hiders(self):
        # in case the hider has been captured (seeker on top of hider or
        # hider on top of seeker), remove it from the list of agents
        if self.hiders[0].captured:
            n_collisions = self.hiders[0].n_collisions
            self.hiders[0].n_collisions = 0
            self.hiders.pop(0)
        else:
            n_collisions = self.hiders[0].n_collisions
            self.hiders[0].n_collisions = 0
        return n_collisions

    # rendering game
    def view(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.draw_grid() # redrawing same grid
        self.draw_walls() # redrawing same walls
        self.draw_seekers() # redrawing new seekers pos
        self.draw_hiders() # redrawing new hiders pos

        #time.sleep(.5) # debug
        self.clock.tick(FPS)
        pygame.display.update()

#-------------------------------------------------------------------------------
""" Seeker class """
class Seeker:
    def __init__(self, id, walls, hiders):
        self.id = id
        self.x = random.randrange(0, WINDOW_WIDTH, BLOCKSIZE)
        self.y = random.randrange(0, WINDOW_HEIGHT, BLOCKSIZE)
        self.rotation = Rotation.UP
        self.colour = YELLOW
        self.size = BLOCKSIZE
        self.n_collisions = 0
        self.angle = 1 # can be 1, -1 (front radar, back radar) or 90 and -90 (right radar, left radar)
        self.radar = [] # for aesthetics
        self.radar_size = 2 # var that controls both the radar (aesthetic) and the rays (effective radar) size
        self.reward = 0
        self.rays_intersections = [] # rays intersections (for visualisation and dist calculations)
        self.rays_origin = None
        self.show_rays = SHOW_RADAR_RAYS # showing rays or not, testing
        self.hider_spotted = False # to give reward only once if the hider is in his radar
        self.rays_endpoints = [] # end points
        self.moved = False

        # observation space data
        self.rays_angles = [] # we could have used the rays_endpoints to create the obs space, this var is just for readibility
        self.intersection_distances = []
        self.intersection_types = []
        self.agent_data = []

        # avoidining spawn on walls
        self.spawn_no_walls(walls)
        # initialising body
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)

        # called here so we have an initial observation space
        # after every move, check radar/intersections for our observation space
        self.create_rays()
        self.find_rays_intersections(walls, hiders)

    # general/spawning functions -----------------------------------------------

    # check end of game (e.g, if seeker rect and hider rect overlap)
    def __eq__(self, other):
        if self.rect.x == other.rect.x and self.rect.y == other.rect.y:
            return True
        return False

    def spawn_no_walls(self, walls):
        # avoiding spawn on walls
        all_walls_coords = []
        for wall in walls:
                all_walls_coords.append((wall.x, wall.y))
        peep_coord = (self.x, self.y)
        while peep_coord in all_walls_coords:
            self.x = random.randrange(0, WINDOW_WIDTH, BLOCKSIZE)
            self.y = random.randrange(0, WINDOW_HEIGHT, BLOCKSIZE)
            peep_coord = (self.x, self.y)

    # utility functions --------------------------------------------------------

    # (37, 2), each row "sees" approx 5 degrees (radians) (rays/radar) and returns
    #   1) the first object met in that direction (0 = EMPTY, 1 = WALL, 2 = HIDER(s))
    #   2) its distance from the origin of the ray/radar
    #
    # # e.g.,  [150. 1]
    # not used, future team MA implementation
    def get_obs_data(self):
        for i in range(0, len(self.rays_angles)): # 37 rays in total
            # could be done in a shorter amount of lines, done for readibility
            obj_intersection_distance = self.intersection_distances[i]
            obj_intersected = self.intersection_types[i]

            # single line
            ray_data = [obj_intersection_distance, obj_intersected]
            self.agent_data.append(ray_data)

    # performing move based on action
    def move(self, walls, hiders, move=None):

        self.moved = False

        self.colour = YELLOW # resetting colour

        # cleaning up reward storage for each step
        self.reset_reward()

        # in case of collision
        previous_x = self.rect.x
        previous_y = self.rect.y

        # apply temporary move
        dmove = move*BLOCKSIZE

        # 0 up (1), 1 right (90), 2 down (-1), 3 left (-90)
        if self.rotation == Rotation.UP: # goes straight (up)
            self.angle = 1
            self.rect.x += 0
            self.rect.y -= dmove
        elif self.rotation == Rotation.RIGHT: # goes right
            self.angle = 90
            self.rect.x += dmove
            self.rect.y += 0
        elif self.rotation == Rotation.DOWN: # goes down
            self.angle = -1
            self.rect.x += 0
            self.rect.y += dmove
        elif self.rotation == Rotation.LEFT: # goes left
            self.angle = -90
            self.rect.x -= dmove
            self.rect.y += 0

        # check if we are out of bounds
        if self.rect.x < 0:
            self.rect.x = 0
            self.reward -= 2
            self.colour = COLLISION_COLOUR
            self.n_collisions += 1
        elif self.rect.x > WINDOW_WIDTH - self.size:
            self.rect.x = WINDOW_WIDTH - self.size
            self.reward -= 2
            self.colour = COLLISION_COLOUR
            self.n_collisions += 1
        if self.rect.y < 0:
            self.rect.y = 0
            self.reward -= 2
            self.colour = COLLISION_COLOUR
            self.n_collisions += 1
        elif self.rect.y > WINDOW_HEIGHT - self.size:
            self.rect.y = WINDOW_HEIGHT - self.size
            self.reward -= 2
            self.colour = COLLISION_COLOUR
            self.n_collisions += 1

        # also check walls collision
        for wall in walls:
            if wall.width == BLOCKSIZE: # we check only the internal walls, not the boundary one
                if self.rect.colliderect(wall):
                    self.rect.x = previous_x
                    self.rect.y = previous_y
                    self.reward -= 2
                    self.colour = COLLISION_COLOUR
                    self.n_collisions += 1

        # check if he moved, for reward calculation
        if self.rect.x == previous_x and self.rect.y == previous_y:
            self.moved = False
        else:
            self.moved = True

        # after every move, check radar/intersections for our observation space
        self.create_rays()
        self.find_rays_intersections(walls, hiders)
        self.check_if_hider_within_radar()

        # the faster seeker finds hider, the bigger his reward
        self.reward -= self.calculate_distance_travelled(previous_x, previous_y)

    def calculate_distance_travelled(self, previous_x, previous_y):

        current_x = self.rect.x
        current_y = self.rect.y
        dist = math.sqrt((previous_x-current_x)**2 + (previous_y-current_y)**2)

        return dist/BLOCKSIZE # 1

    # verify whether opponent is inside personal radar
    def check_if_hider_within_radar(self):

        all_possibilities = []
        all_possibilitiestmp = []

        # if hider was already in his vision, don't reward, but check if he's still there
        if(self.hider_spotted):

            for idx, type in enumerate(self.intersection_types):
                if type == 2:
                    all_possibilitiestmp.append(self.intersection_distances[idx])

            if len(all_possibilitiestmp) == 0:
                self.hider_spotted = False
                # if seeker run away from hider
                self.reward -= 5
            else:
                min_dist_to_hider = min(all_possibilitiestmp)

                # if seeker got even closer to hider
                if min_dist_to_hider <= BLOCKSIZE:
                    self.reward += 2.5

        else:
            # if hider is within seeker radar
            for idx, type in enumerate(self.intersection_types):
                if type == 2:
                    all_possibilities.append(self.intersection_distances[idx])

            if len(all_possibilities) >= 1:
                min_dist_to_hider = min(all_possibilities)

                if min_dist_to_hider <= BLOCKSIZE*2:
                    self.hider_spotted = True
                    self.reward += 0.5

                elif min_dist_to_hider > BLOCKSIZE*2:
                    self.hider_spotted = True
                    self.reward += 0.25

    # resetting internal reward util
    def reset_reward(self):
        self.reward = 0

    # returning reward util
    def get_reward(self):
        return self.reward

    # creating the rays (effective radar) - in their standard position
    def create_rays(self):
        if(self.angle == 1):
            self.rays_endpoints = self.create_rays_up()
            #self.rays_origin = self.rect.midbottom # problem with "parallel" lines, workoud -> start from center of block
            self.rays_origin = self.rect.center
        elif(self.angle == 90):
            self.rays_endpoints = self.create_rays_right()
            #self.rays_origin = self.rect.midleft # problem with "parallel" lines, workoud -> start from center of block
            self.rays_origin = self.rect.center
        elif(self.angle == -1):
            self.rays_endpoints = self.create_rays_down()
            #self.rays_origin = self.rect.midtop # problem with "parallel" lines, workoud -> start from center of block
            self.rays_origin = self.rect.center
        elif(self.angle == -90):
            self.rays_endpoints = self.create_rays_left()
            #self.rays_origin = self.rect.midright # problem with "parallel" lines, workoud -> start from center of block
            self.rays_origin = self.rect.center

    # effective radar, 4 functions for code readibility (functionality a bit off), the angles change
    # since the "vision" (rays) start from the center, we make them smaller in order to fit radar (aesthetics) and the radar size
    def create_rays_up(self):
        self.rays_angles = [math.radians(angle) for angle in range(-180, 5, 5)]
        return [(self.rect.centerx + (self.radar_size+1)*self.size * math.cos(math.radians(angle)), self.rect.centery + (self.radar_size+1)*self.size * math.sin(math.radians(angle))) for angle in range(-180, 5, 5)]

    def create_rays_right(self):
        self.rays_angles = [math.radians(angle) for angle in range(-90, 95, 5)]
        return [(self.rect.centerx + (self.radar_size+1)*self.size * math.cos(math.radians(angle)), self.rect.centery + (self.radar_size+1)*self.size * math.sin(math.radians(angle))) for angle in range(-90, 95, 5)]

    def create_rays_down(self):
        self.rays_angles = [math.radians(angle) for angle in range(0, 185, 5)]
        return [(self.rect.centerx + (self.radar_size+1)*self.size * math.cos(math.radians(angle)), self.rect.centery + (self.radar_size+1)*self.size * math.sin(math.radians(angle))) for angle in range(0, 185, 5)]

    def create_rays_left(self):
        self.rays_angles = [math.radians(angle) for angle in range(90, 275, 5)]
        return [(self.rect.centerx + (self.radar_size+1)*self.size * math.cos(math.radians(angle)), self.rect.centery + (self.radar_size+1)*self.size * math.sin(math.radians(angle))) for angle in range(90, 275, 5)]

    # 37 rays (180 degrees in total)
    def find_rays_intersections(self, walls, hiders):
        # http://www.jeffreythompson.org/collision-detection/line-rect.php

        # each square has 4 lines, we check for intersections with all of them
        # topleft, topright
        # topright, bottomright
        # bottomright, bottomleft
        # bottomleft, topleft

        # we first "transform" the boxes into lines
        # we put everything in one list
        obstacles = []

        for hider in hiders:
            obstacles.append(((hider.rect.topleft), (hider.rect.topright), 2))
            obstacles.append(((hider.rect.topright), (hider.rect.bottomright), 2))
            obstacles.append(((hider.rect.bottomright), (hider.rect.bottomleft), 2))
            obstacles.append(((hider.rect.bottomleft), (hider.rect.topleft), 2))


        for wall in walls:
            obstacles.append(((wall.topleft), (wall.topright), 1))
            obstacles.append(((wall.topright), (wall.bottomright), 1))
            obstacles.append(((wall.bottomright), (wall.bottomleft), 1))
            obstacles.append(((wall.bottomleft), (wall.topleft), 1))

        for endpoint in self.rays_endpoints:
            # max dist here is just a number to "check" the full length of the ray (for intersections)
            endpoint, maxdist, obs_type = self.line_intersections(obstacles, self.rays_origin, endpoint)

            # calculating distance from start of vision to max rays length
            # dist in this case is the effective distance from rays origin to intersection (if there are any)
            dist = self.calculate_rays_distances_and_hits(self.rays_origin, endpoint)

            # data for our observation space
            self.intersection_distances.append(dist)
            self.intersection_types.append(obs_type)

            # update the rays_endpoints with intersections (for visualisation)
            self.rays_intersections.append(endpoint)

        # changing rays endpoints if there are intersections
        #self.rays_endpoints = self.rays_intersections.copy() # CHANGE: we keep the original endpoints intact for our observation space

    def calculate_rays_distances_and_hits(self, origin, endpoint):

        x1 = origin[0]
        y1 = origin[1]
        x2 = endpoint[0]
        y2 = endpoint[1]

        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2) # calculating distance in every case (hit or non hit)

        return dist

    # references/sources:
    # https://stackoverflow.com/questions/56316263/problem-with-finding-the-closest-intersection/56316370#56316370
    # https://stackoverflow.com/questions/56312503/problem-with-calculating-line-intersections
    # https://github.com/Rabbid76/PyGameExamplesAndAnswers/blob/master/documentation/pygame/pygame_collision_and_intesection.md#line-and-line
    def line_intersections(self, obstacles, start_ray, end_ray):

        maxdist = 10000000000000000000
        endpoint = end_ray

        new_obs_type = 0 # empty standard objecr
        for start_obs, end_obs, obs_type in obstacles:
            d = (end_ray[0]-start_ray[0]) * (end_obs[1]-start_obs[1]) + (end_ray[1]-start_ray[1]) * (start_obs[0]-end_obs[0])

            if d != 0:
                t = ((start_obs[0]-start_ray[0]) * (end_obs[1]-start_obs[1]) + (start_obs[1]-start_ray[1]) * (start_obs[0]-end_obs[0])) / d
                u = ((start_obs[0]-start_ray[0]) * (end_ray[1]-start_ray[1]) + (start_obs[1]-start_ray[1]) * (start_ray[0]-end_ray[0])) / d
                if 0 <= t <= 1 and 0 <= u <= 1:
                    vx, vy = (end_ray[0]-start_ray[0]) * t, (end_ray[1]-start_ray[1]) * t
                    dist = vx*vx + vy*vy
                    if dist < maxdist:
                        px, py = (end_obs[0] * u + start_obs[0] * (1-u)), (end_obs[1] * u + start_obs[1] * (1-u))
                        maxdist = dist
                        endpoint = (px, py)
                        new_obs_type = obs_type # if there's an intersection, change obs type

        return endpoint, maxdist, new_obs_type

    # clearing radars after every draw call
    def clear_radar(self):
        self.rays_angles.clear() # not needed but just to be sure
        self.rays_endpoints.clear()
        self.rays_intersections.clear()
        self.intersection_distances.clear()
        self.intersection_types.clear()
        self.radar.clear()

    # clearing obs data
    def clear_agent_data(self):
        self.agent_data.clear()

    # drawing functions --------------------------------------------------------

    def draw(self, display, walls):
        # left, top, width, height
        pygame.draw.rect(display, self.colour, self.rect)
        self.draw_radar(display, walls) # aesthetics only pretty much

        # drawing effective radar (rays) - debug
        if self.show_rays:
            self.draw_rays(display)

    def draw_rays(self, display):

        for endpoint in self.rays_intersections:
            pygame.draw.line(display, BLACK, self.rays_origin, endpoint)
            pygame.draw.circle(display, (0, 255, 0), (round(endpoint[0]), round(endpoint[1])), 3)

    # for aesthetics only
    def draw_radar(self, display, walls):
        # draw radar
        # front and back radars (1, -1)
        if (self.angle == -1 or self.angle == 1):
            # "subradars"
            # radar left
            for i in range(1, self.radar_size+1):
                radar_left = pygame.Rect(self.rect.x - self.size*i*self.angle, self.rect.y, self.size, self.size)
                pygame.gfxdraw.box(display, radar_left, TRANSPARENT_RADAR)
                self.radar.append(radar_left)
                # radar right
                radar_right = pygame.Rect(self.rect.x + self.size*i*self.angle, self.rect.y, self.size, self.size)
                pygame.gfxdraw.box(display, radar_right, TRANSPARENT_RADAR)
                self.radar.append(radar_right)
                # radar front
                radar_front = pygame.Rect(self.rect.x, self.rect.y - self.size*i*self.angle, self.size, self.size)
                pygame.gfxdraw.box(display, radar_front, TRANSPARENT_RADAR)
                self.radar.append(radar_front)

                for j in range(1, self.radar_size+1):
                    radar_front_left = pygame.Rect(radar_left.x, radar_left.top - self.size*j*self.angle, self.size, self.size)
                    radar_front_right = pygame.Rect(radar_right.x, radar_right.top - self.size*j*self.angle, self.size, self.size)
                    pygame.gfxdraw.box(display, radar_front_left, TRANSPARENT_RADAR)
                    pygame.gfxdraw.box(display, radar_front_right, TRANSPARENT_RADAR)
                    self.radar.append(radar_front_left)
                    self.radar.append(radar_front_right)


            for wall in walls:
                if wall.width == BLOCKSIZE: # we check only the internal walls, not the boundary one
                    for idx, radar in enumerate(self.radar):
                        if radar.colliderect(wall):
                            pygame.gfxdraw.box(display, radar, BLACK)
                            self.radar.pop(idx)

        # right or left (90 or -90)
        elif(self.angle == 90 or self.angle == -90):
            tmp_angle = self.angle/90
            # "subradars"
            # radar left
            for i in range(1, self.radar_size+1):
                radar_left = pygame.Rect(self.rect.x, self.rect.y - self.size*i*tmp_angle, self.size, self.size)
                pygame.gfxdraw.box(display, radar_left, TRANSPARENT_RADAR)
                self.radar.append(radar_left)
                # radar right
                radar_right = pygame.Rect(self.rect.x, self.rect.y + self.size*i*tmp_angle, self.size, self.size)
                pygame.gfxdraw.box(display, radar_right, TRANSPARENT_RADAR)
                self.radar.append(radar_right)
                # radar front
                radar_front = pygame.Rect(self.rect.x + self.size*i*tmp_angle, self.rect.y , self.size, self.size)
                pygame.gfxdraw.box(display, radar_front, TRANSPARENT_RADAR)
                self.radar.append(radar_front)

                for j in range(1, self.radar_size+1):
                    radar_front_left = pygame.Rect(radar_left.x + self.size*j*tmp_angle, radar_left.y, self.size, self.size)
                    radar_front_right = pygame.Rect(radar_right.x + self.size*j*tmp_angle, radar_right.y, self.size, self.size)
                    pygame.gfxdraw.box(display, radar_front_left, TRANSPARENT_RADAR)
                    pygame.gfxdraw.box(display, radar_front_right, TRANSPARENT_RADAR)
                    self.radar.append(radar_front_left)
                    self.radar.append(radar_front_right)


            for wall in walls:
                if wall.width == BLOCKSIZE: # we check only the internal walls, not the boundary one
                    for idx, radar in enumerate(self.radar):
                        if radar.colliderect(wall):
                            pygame.gfxdraw.box(display, radar, BLACK)
                            self.radar.pop(idx)
""" End of Seeker class """

#-------------------------------------------------------------------------------
""" Hider class """
class Hider:
    def __init__(self, id, walls, seekers):
        self.id = id
        self.x = random.randrange(0, WINDOW_WIDTH, BLOCKSIZE)
        self.y = random.randrange(0, WINDOW_HEIGHT, BLOCKSIZE)
        self.captured = False
        self.rotation = Rotation.UP
        self.colour = TURQUOISE
        self.size = BLOCKSIZE
        self.n_collisions = 0 # testing
        self.angle = 1 # can be 1, -1 (front radar, back radar) or 90 and -90 (right radar, left radar)
        self.radar = [] # for aesthetics
        self.radar_size = 2 # var that controls both the radar (aesthetic) and the rays (effective radar) size
        self.reward = 0
        self.rays_intersections = [] # rays intersections (for visualisation and dist calculations)
        self.rays_origin = None
        self.show_rays = SHOW_RADAR_RAYS # showing rays or not, testing
        self.seeker_spotted = False # to give reward only once if the hider is in his radar
        self.rays_endpoints = [] # end points
        self.moved = False

        # observation space data
        self.rays_angles = [] # we could have used the rays_endpoints to create the obs space, this var is just for readibility
        self.intersection_distances = []
        self.intersection_types = []
        self.agent_data = []

        # avoidining spawn on walls
        self.spawn_no_walls(walls)
        # initialising body
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)

        # called here so we have an initial observation space
        # after every move, check radar/intersections for our observation space
        self.create_rays()
        self.find_rays_intersections(walls, seekers)

    # general/spawning functions -----------------------------------------------

    # check end of game (e.g, if seeker rect and hider rect overlap)
    def __eq__(self, other):
        if self.rect.x == other.rect.x and self.rect.y == other.rect.y:
            return True
        return False

    # avoiding spawn on walls
    def spawn_no_walls(self, walls):
        all_walls_coords = []
        for wall in walls:
                all_walls_coords.append((wall.x, wall.y))
        peep_coord = (self.x, self.y)
        while peep_coord in all_walls_coords:
            self.x = random.randrange(0, WINDOW_WIDTH, BLOCKSIZE)
            self.y = random.randrange(0, WINDOW_HEIGHT, BLOCKSIZE)
            peep_coord = (self.x, self.y)

    # utility functions --------------------------------------------------------

    # (37, 2), each row "sees" approx 5 degrees (radians) (rays/radar) and returns
    #   1) the first object met in that direction (0 = EMPTY, 1 = WALL, 2 = HIDER(s))
    #   2) its distance from the origin of the ray/radar
    #
    # # e.g.,  [150. 1]
    # not used, future team MA implementation
    def get_obs_data(self):
        for i in range(0, len(self.rays_angles)): # 37 rays in total
            # could be done in a shorter amount of lines, done for readibility
            obj_intersection_distance = self.intersection_distances[i]
            obj_intersected = self.intersection_types[i]

            # single line
            ray_data = [obj_intersection_distance, obj_intersected]
            self.agent_data.append(ray_data)

    # performing move based on action
    def move(self, walls, seekers, move=None):

        self.moved = False

        self.colour = TURQUOISE

        # cleaning up reward storage for each step
        self.reset_reward()

        # in case of collision
        previous_x = self.rect.x
        previous_y = self.rect.y

        # apply temporary move
        dmove = move*BLOCKSIZE

        # 0 up (1), 1 right (90), 2 down (-1), 3 left (-90)
        if self.rotation == Rotation.UP: # goes straight (up)
            self.angle = 1
            self.rect.x += 0
            self.rect.y -= dmove
        elif self.rotation == Rotation.RIGHT: # goes right
            self.angle = 90
            self.rect.x += dmove
            self.rect.y += 0
        elif self.rotation == Rotation.DOWN: # goes down
            self.angle = -1
            self.rect.x += 0
            self.rect.y += dmove
        elif self.rotation == Rotation.LEFT: # goes left
            self.angle = -90
            self.rect.x -= dmove
            self.rect.y += 0

        # check if we are out of bounds
        if self.rect.x < 0:
            self.rect.x = 0
            self.reward -= 2
            self.colour = COLLISION_COLOUR
            self.n_collisions += 1
        elif self.rect.x > WINDOW_WIDTH - self.size:
            self.rect.x = WINDOW_WIDTH - self.size
            self.reward -= 2
            self.colour = COLLISION_COLOUR
            self.n_collisions += 1
        if self.rect.y < 0:
            self.rect.y = 0
            self.reward -= 2
            self.colour = COLLISION_COLOUR
            self.n_collisions += 1
        elif self.rect.y > WINDOW_HEIGHT - self.size:
            self.rect.y = WINDOW_HEIGHT - self.size
            self.reward -= 2
            self.colour = COLLISION_COLOUR
            self.n_collisions += 1

        # also check walls collision
        for wall in walls:
            if wall.width == BLOCKSIZE: # we check only the internal walls, not the boundary one
                if self.rect.colliderect(wall):
                    self.rect.x = previous_x
                    self.rect.y = previous_y
                    self.reward -= 2
                    self.colour = COLLISION_COLOUR
                    self.n_collisions += 1


        # check if he moved, for reward calculation
        if self.rect.x == previous_x and self.rect.y == previous_y:
            self.moved = False
        else:
            self.moved = True

        # after every move, check radar/intersections for our observation space
        self.create_rays()
        self.find_rays_intersections(walls, seekers)
        self.check_if_seeker_within_radar()

        # the more hider moves, the bigger its reward
        self.reward += self.calculate_distance_travelled(previous_x, previous_y)

    def calculate_distance_travelled(self, previous_x, previous_y):

        current_x = self.rect.x
        current_y = self.rect.y
        dist = math.sqrt((previous_x-current_x)**2 + (previous_y-current_y)**2)

        return dist/BLOCKSIZE

    # verify if opponent is inside personal radar
    def check_if_seeker_within_radar(self):

        all_possibilities = []
        all_possibilitiestmp = []

        # if seeker was already in his vision, don't reward, but check if he's still there
        if(self.seeker_spotted):

            for idx, type in enumerate(self.intersection_types):
                if type == 2:
                    all_possibilitiestmp.append(self.intersection_distances[idx])

            if len(all_possibilitiestmp) == 0:
                self.seeker_spotted = False
                # if hider run away from seeker
                self.reward += 5
            else:
                min_dist_to_seeker = min(all_possibilitiestmp)

                # if hider got closer to seeker
                if min_dist_to_seeker <= BLOCKSIZE:
                    self.reward -= 2.5
        else:
            # if seeker is within hider radar
            for idx, type in enumerate(self.intersection_types):
                if type == 2:
                    all_possibilities.append(self.intersection_distances[idx])

            if len(all_possibilities) >= 1:
                min_dist_to_seeker = min(all_possibilities)

                if min_dist_to_seeker <= BLOCKSIZE*2:
                    self.seeker_spotted = True
                    self.reward -= 0.5
                elif min_dist_to_seeker > BLOCKSIZE*2:
                    self.seeker_spotted = True
                    self.reward -= 0.25

    # reset internal reward util
    def reset_reward(self):
        self.reward = 0

    # return internal reward util
    def get_reward(self):
        return self.reward

    # creating the rays (effective radar) - in their standard position
    def create_rays(self):
        if(self.angle == 1):
            self.rays_endpoints = self.create_rays_up()
            #self.rays_origin = self.rect.midbottom # problem with "parallel" lines, workoud -> start from center of block
            self.rays_origin = self.rect.center
        elif(self.angle == 90):
            self.rays_endpoints = self.create_rays_right()
            #self.rays_origin = self.rect.midleft # problem with "parallel" lines, workoud -> start from center of block
            self.rays_origin = self.rect.center
        elif(self.angle == -1):
            self.rays_endpoints = self.create_rays_down()
            #self.rays_origin = self.rect.midtop # problem with "parallel" lines, workoud -> start from center of block
            self.rays_origin = self.rect.center
        elif(self.angle == -90):
            self.rays_endpoints = self.create_rays_left()
            #self.rays_origin = self.rect.midright # problem with "parallel" lines, workoud -> start from center of block
            self.rays_origin = self.rect.center

    # effective radar, 4 functions for code readibility (functionality a bit off), the angles change
    # since the "vision" (rays) start from the center, we make them smaller in order to fit radar (aesthetics) and the radar size
    def create_rays_up(self):
        self.rays_angles = [math.radians(angle) for angle in range(-180, 5, 5)]
        return [(self.rect.centerx + (self.radar_size+1)*self.size * math.cos(math.radians(angle)), self.rect.centery + (self.radar_size+1)*self.size * math.sin(math.radians(angle))) for angle in range(-180, 5, 5)]

    def create_rays_right(self):
        self.rays_angles = [math.radians(angle) for angle in range(-90, 95, 5)]
        return [(self.rect.centerx + (self.radar_size+1)*self.size * math.cos(math.radians(angle)), self.rect.centery + (self.radar_size+1)*self.size * math.sin(math.radians(angle))) for angle in range(-90, 95, 5)]

    def create_rays_down(self):
        self.rays_angles = [math.radians(angle) for angle in range(0, 185, 5)]
        return [(self.rect.centerx + (self.radar_size+1)*self.size * math.cos(math.radians(angle)), self.rect.centery + (self.radar_size+1)*self.size * math.sin(math.radians(angle))) for angle in range(0, 185, 5)]

    def create_rays_left(self):
        self.rays_angles = [math.radians(angle) for angle in range(90, 275, 5)]
        return [(self.rect.centerx + (self.radar_size+1)*self.size * math.cos(math.radians(angle)), self.rect.centery + (self.radar_size+1)*self.size * math.sin(math.radians(angle))) for angle in range(90, 275, 5)]

    # 37 rays (180 degrees in total)
    def find_rays_intersections(self, walls, seekers):

        # http://www.jeffreythompson.org/collision-detection/line-rect.php
        # each square has 4 lines, we check for intersections with all of them
        # topleft, topright
        # topright, bottomright
        # bottomright, bottomleft
        # bottomleft, topleft

        # we first "transform" the boxes into lines
        # we put everything in one list
        obstacles = []

        for seeker in seekers:
            obstacles.append(((seeker.rect.topleft), (seeker.rect.topright), 2))
            obstacles.append(((seeker.rect.topright), (seeker.rect.bottomright), 2))
            obstacles.append(((seeker.rect.bottomright), (seeker.rect.bottomleft), 2))
            obstacles.append(((seeker.rect.bottomleft), (seeker.rect.topleft), 2))


        for wall in walls:
            obstacles.append(((wall.topleft), (wall.topright), 1))
            obstacles.append(((wall.topright), (wall.bottomright), 1))
            obstacles.append(((wall.bottomright), (wall.bottomleft), 1))
            obstacles.append(((wall.bottomleft), (wall.topleft), 1))

        for endpoint in self.rays_endpoints:
            # max dist here is just a number to "check" the full length of the ray (for intersections)
            endpoint, maxdist, obs_type = self.line_intersections(obstacles, self.rays_origin, endpoint)

            # calculating distance from start of vision to max rays length
            # dist in this case is the effective distance from rays origin to intersection (if there are any)
            dist = self.calculate_rays_distances_and_hits(self.rays_origin, endpoint)

            # data for our observation space
            self.intersection_distances.append(dist)
            self.intersection_types.append(obs_type)

            # update the rays_endpoints with intersections (for visualisation)
            self.rays_intersections.append(endpoint)

        # changing rays endpoints if there are intersections
        #self.rays_endpoints = self.rays_intersections.copy() # CHANGE: we keep the original endpoints intact for our observation space

    def calculate_rays_distances_and_hits(self, origin, endpoint):

        x1 = origin[0]
        y1 = origin[1]
        x2 = endpoint[0]
        y2 = endpoint[1]

        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2) # calculating distance in every case (hit or non hit)

        return dist

    # References/sources:
    # https://stackoverflow.com/questions/56316263/problem-with-finding-the-closest-intersection/56316370#56316370
    # https://stackoverflow.com/questions/56312503/problem-with-calculating-line-intersections
    # https://github.com/Rabbid76/PyGameExamplesAndAnswers/blob/master/documentation/pygame/pygame_collision_and_intesection.md#line-and-line
    def line_intersections(self, obstacles, start_ray, end_ray):

        maxdist = 10000000000000000000
        endpoint = end_ray

        new_obs_type = 0 # empty standard objecr
        for start_obs, end_obs, obs_type in obstacles:
            d = (end_ray[0]-start_ray[0]) * (end_obs[1]-start_obs[1]) + (end_ray[1]-start_ray[1]) * (start_obs[0]-end_obs[0])

            if d != 0:
                t = ((start_obs[0]-start_ray[0]) * (end_obs[1]-start_obs[1]) + (start_obs[1]-start_ray[1]) * (start_obs[0]-end_obs[0])) / d
                u = ((start_obs[0]-start_ray[0]) * (end_ray[1]-start_ray[1]) + (start_obs[1]-start_ray[1]) * (start_ray[0]-end_ray[0])) / d
                if 0 <= t <= 1 and 0 <= u <= 1:
                    vx, vy = (end_ray[0]-start_ray[0]) * t, (end_ray[1]-start_ray[1]) * t
                    dist = vx*vx + vy*vy
                    if dist < maxdist:
                        px, py = (end_obs[0] * u + start_obs[0] * (1-u)), (end_obs[1] * u + start_obs[1] * (1-u))
                        maxdist = dist
                        endpoint = (px, py)
                        new_obs_type = obs_type # if there's an intersection, change obs type

        return endpoint, maxdist, new_obs_type

    # clearing radars after every draw call
    def clear_radar(self):
        self.rays_angles.clear() # not needed but just to be sure
        self.rays_endpoints.clear()
        self.rays_intersections.clear()
        self.intersection_distances.clear()
        self.intersection_types.clear()
        self.radar.clear()

    # clearing obs data
    def clear_agent_data(self):
        self.agent_data.clear()

    # drawing functions --------------------------------------------------------

    def draw(self, display, walls):
        # left, top, width, height
        pygame.draw.rect(display, self.colour, self.rect)
        #self.draw_radar(display, walls) # aesthetics only pretty much - not shown for hider

        # drawing effective radar (rays) - debug
        if self.show_rays:
            self.draw_rays(display)

    def draw_rays(self, display):

        for endpoint in self.rays_intersections:
            pygame.draw.line(display, TURQUOISE, self.rays_origin, endpoint)
            pygame.draw.circle(display, (0, 255, 0), (round(endpoint[0]), round(endpoint[1])), 3)

    # for aesthetics only
    def draw_radar(self, display, walls):
        # draw radar
        # front and back radars (1, -1)
        if (self.angle == -1 or self.angle == 1):
            # "subradars"
            # radar left
            for i in range(1, self.radar_size+1):
                radar_left = pygame.Rect(self.rect.x - self.size*i*self.angle, self.rect.y, self.size, self.size)
                pygame.gfxdraw.box(display, radar_left, TRANSPARENT_RADAR)
                self.radar.append(radar_left)
                # radar right
                radar_right = pygame.Rect(self.rect.x + self.size*i*self.angle, self.rect.y, self.size, self.size)
                pygame.gfxdraw.box(display, radar_right, TRANSPARENT_RADAR)
                self.radar.append(radar_right)
                # radar front
                radar_front = pygame.Rect(self.rect.x, self.rect.y - self.size*i*self.angle, self.size, self.size)
                pygame.gfxdraw.box(display, radar_front, TRANSPARENT_RADAR)
                self.radar.append(radar_front)

                for j in range(1, self.radar_size+1):
                    radar_front_left = pygame.Rect(radar_left.x, radar_left.top - self.size*j*self.angle, self.size, self.size)
                    radar_front_right = pygame.Rect(radar_right.x, radar_right.top - self.size*j*self.angle, self.size, self.size)
                    pygame.gfxdraw.box(display, radar_front_left, TRANSPARENT_RADAR)
                    pygame.gfxdraw.box(display, radar_front_right, TRANSPARENT_RADAR)
                    self.radar.append(radar_front_left)
                    self.radar.append(radar_front_right)


            for wall in walls:
                if wall.width == BLOCKSIZE: # we check only the internal walls, not the boundary one
                    for idx, radar in enumerate(self.radar):
                        if radar.colliderect(wall):
                            pygame.gfxdraw.box(display, radar, BLACK)
                            self.radar.pop(idx)

        # right or left (90 or -90)
        elif(self.angle == 90 or self.angle == -90):
            tmp_angle = self.angle/90
            # "subradars"
            # radar left
            for i in range(1, self.radar_size+1):
                radar_left = pygame.Rect(self.rect.x, self.rect.y - self.size*i*tmp_angle, self.size, self.size)
                pygame.gfxdraw.box(display, radar_left, TRANSPARENT_RADAR)
                self.radar.append(radar_left)
                # radar right
                radar_right = pygame.Rect(self.rect.x, self.rect.y + self.size*i*tmp_angle, self.size, self.size)
                pygame.gfxdraw.box(display, radar_right, TRANSPARENT_RADAR)
                self.radar.append(radar_right)
                # radar front
                radar_front = pygame.Rect(self.rect.x + self.size*i*tmp_angle, self.rect.y , self.size, self.size)
                pygame.gfxdraw.box(display, radar_front, TRANSPARENT_RADAR)
                self.radar.append(radar_front)

                for j in range(1, self.radar_size+1):
                    radar_front_left = pygame.Rect(radar_left.x + self.size*j*tmp_angle, radar_left.y, self.size, self.size)
                    radar_front_right = pygame.Rect(radar_right.x + self.size*j*tmp_angle, radar_right.y, self.size, self.size)
                    pygame.gfxdraw.box(display, radar_front_left, TRANSPARENT_RADAR)
                    pygame.gfxdraw.box(display, radar_front_right, TRANSPARENT_RADAR)
                    self.radar.append(radar_front_left)
                    self.radar.append(radar_front_right)


            for wall in walls:
                if wall.width == BLOCKSIZE: # we check only the internal walls, not the boundary one
                    for idx, radar in enumerate(self.radar):
                        if radar.colliderect(wall):
                            pygame.gfxdraw.box(display, radar, BLACK)
                            self.radar.pop(idx)
""" End of Hider class """
