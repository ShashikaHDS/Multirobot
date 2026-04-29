import gym
from gym import spaces
import numpy as np
import cv2
import random
import time
from collections import deque
import random
import matplotlib.pyplot as plt
import math
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point
import pygame
import pygame.image
import sys
import numpy as np
import math
import pandas as pd
import csv

bg_color = (255,255,255)
area_color = (255,255,0)
offsetcolor = (255,255,0)
area_color = pygame.Color("orange")
rob_color = (0,0,250)
rob1_color = (0,0,250)
rob2_color = (0,0,200)
rob3_color = (0,0,150)
rob4_color = (0,0,100)
color = (0,0,0)
grid_color = pygame.Color("grey")
width = 400
height = 400
cells = []
cell_size = 25
pygame.init()
screen = pygame.display.set_mode((width,height))
#screen = pygame.Surface((width,height))
for y in range(height//cell_size):
    row = []
    for x in range(width // cell_size):
        rect = pygame.Rect(x*cell_size, y*cell_size, cell_size,cell_size)
        row.append(rect)
    cells.append(row)

def computeOrigins(pos,q, linkLen):
    # Computes o, the (x, y) coordinate of the DH frame for each link in q
    l = 24*np.sqrt(2)/2
    qSum = np.cumsum(q, axis=0)
    qSum = [math.radians(i) for i in qSum]

    p = [[pos[0],pos[1]],[pos[0],pos[1]],[pos[0],pos[1]],[pos[0],pos[1]]]
    oDelta = np.column_stack((linkLen * np.cos(qSum), linkLen * np.sin(qSum)))
    o = p + np.cumsum(oDelta, axis=0)
    o = np.insert(o, 0, [pos[0],pos[1]], axis=0)
    
    return o

def module_pos(o, q,linkLen2):
    all_pos = [[],[],[],[]]
    qsum = np.cumsum(q, axis=0)
    
    qsum = [math.radians(i) for i in qsum]
    qsum1 = [j - np.pi / 4 for j in qsum]
    
    for w in range(len(q)):
        
        if w == 2:
            all_pos[w] = o[w] + np.array([[0, 0],[linkLen2 * np.cos(qsum1[w]), linkLen2 * np.sin(qsum1[w])],
                                          [linkLen2 * (np.cos(qsum1[w])- np.sin(qsum1[w])),linkLen2 * (np.sin(qsum1[w])+np.cos(qsum1[w]))],
                                          [linkLen2 * -np.sin(qsum1[w]), linkLen2 * np.cos(qsum1[w])]])
        else:
        
            all_pos[w] = o[w] + np.array([[0, 0],[linkLen2 * np.cos(qsum[w]), linkLen2 * np.sin(qsum[w])],
                                          [linkLen2 * (np.cos(qsum[w])- np.sin(qsum[w])),linkLen2 * (np.sin(qsum[w])+np.cos(qsum[w]))],
                                          [linkLen2 * -np.sin(qsum[w]), linkLen2 * np.cos(qsum[w])]])

    return all_pos

def middle_pos(o, q):
    all_pos = [[],[],[],[]]
    qsum = np.cumsum(q, axis=0)
    l = 24*np.sqrt(2)/2 
    qsum = [math.radians(i) for i in qsum]
    for w in range(len(q)):
        
        if w == 2:
            all_pos[w] = o[w] + [l*np.cos(qsum[w]),l*np.sin(qsum[w])]
        else:
           
            all_pos[w] = o[w] + [l*np.cos(qsum[w]+np.pi/4),l*np.sin(qsum[w]+np.pi/4)]

    return all_pos

def collision(m1,m2,m3,m4,obs, boundaries,all_pos):
    intersect = False

    intersections = []
    for poly_obs in obs:
        #inter1 = m1.intersection(poly_obs)
        #inter2 = m2.intersection(poly_obs)
        #inter3 = m3.intersection(poly_obs)
        #inter4 = m4.intersection(poly_obs)
        #a = [list(inter1.exterior.coords),list(inter2.exterior.coords),list(inter2.exterior.coords),list(inter2.exterior.coords)]
        intersect = intersect or (m1.intersection(poly_obs).area + m2.intersection(poly_obs).area+m3.intersection(poly_obs).area + m4.intersection(poly_obs).area) > 0

        #for cords in a:
        #    if len(cords)>0:
        #        intersections.append(cords)
                
    
    out = False

    
    for m in all_pos:
        for p in m:
            if p[0] < boundaries[0] or p[1]<boundaries[1] or p[0]>boundaries[2] or p[1]>boundaries[3]:
                out = True
    
    s_col = False
    if round(m1.intersection(m2).area,5) > 0 or round(m1.intersection(m3).area,5) > 0 or round(m1.intersection(m4).area,5) > 0 or round(m2.intersection(m3).area,5) > 0 or round(m2.intersection(m4).area,5) > 0 or round(m3.intersection(m4).area,5) > 0:
        s_col = True
    
    return intersect ,out or s_col

def obst_poly(obs):
    obs_poly  = []
    for obst in obs:
        obs_poly.append(Polygon(obst))
    return obs_poly

def calc_distance(start,end):
    distance = np.sum(np.sqrt(np.sum(np.subtract(end,start)**2,)))
    return distance

def calc_distance_module(start,end):
    deltax = abs(end[0] - start[0])
    deltay = abs(end[1] - start[1])
    distance = np.sum(abs(np.subtract(end,start)))
    return distance, deltax, deltay

class SnekEnv(gym.Env):

    def __init__(self):
        super(SnekEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:

        self.WIDTH = 400
        self.HEIGHT = 400
        self.action_space = spaces.MultiDiscrete([3,3,3,3,3,3])
        # Example for using image as input (channel-first; channel-last also works): ?????
        self.observation_space = gym.spaces.Dict({'img': gym.spaces.Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH), dtype=np.uint8),
                                                  'num': gym.spaces.Box(low = 0, high = 500, shape = (6,),dtype=np.int32)})
        
        self.counter = 0
        
        #self.obs = [[(110,-1),(120,-1),(120,45),(110,45)],[(110,75),(120,75),(120,120),(110,120)]]
        self.obs = [[(110,-1),(120,-1),(120,45),(110,45)],[(110,75),(120,75),(120,120),(110,120)]]

        self.linkLen = [24,24,24*math.sqrt(2),24]
        self.linkLen2 = 24
        self.boundaries = (0,0,320,160)

        self.obstacle_polygon = obst_poly(self.obs)
        
        self.observobst = []
        self.count = 0
        self.val = 0
        self.environment = True
        self.z = 0
        self.incre = 1
        self.cumulative = 0

    def step(self, action):
        
        
        self.move = [-1,0,1]
        self.orient = [-5,0,5]
        self.orient1 = [-5,0,5]
        self.pos = [self.pos[0]+self.move[action[0]],self.pos[1] + self.move[action[1]]]
        self.ang[0] += self.orient[action[2]]
        if self.ang[1] + self.orient1[action[3]] <= 0 and self.ang[1] + self.orient1[action[3]] >= -180:
            self.ang[1] += self.orient1[action[3]]
        if self.ang[2] + self.orient1[action[4]] <= 45 and self.ang[2] + self.orient1[action[4]] >= -135:
            self.ang[2] += self.orient1[action[4]]
        if self.ang[3] + self.orient1[action[5]] <= 45 and self.ang[3] + self.orient1[action[5]] >= -135:
            self.ang[3] += self.orient1[action[5]]
        
        posi = computeOrigins(self.pos,self.ang, self.linkLen)
        
        all_pos = module_pos(posi[0:4], self.ang,self.linkLen2)
        
        headx = self.pos[0]
        heady = self.pos[1]
        ang1 = self.ang[0]
        ang2 = self.ang[1]
        ang3 = self.ang[2]
        ang4 = self.ang[3]

        goal_reward = 0
        self.done = False
         

   
        m1 = Polygon(all_pos[0])
        m2 = Polygon(all_pos[1])
        m3 = Polygon(all_pos[2])
        m4 = Polygon(all_pos[3])

        m1off = m1.buffer(distance=24, cap_style=3, join_style=2)
        m2off = m2.buffer(distance=24, cap_style=3, join_style=2)
        m3off = m3.buffer(distance=24, cap_style=3, join_style=2)
        m4off = m4.buffer(distance=24, cap_style=3, join_style=2)



        self.observobst  = []
        for obsp in self.obstacle_polygon:
                inter1 = m1off.intersection(obsp)
                inter2 = m2off.intersection(obsp)
                inter3 = m3off.intersection(obsp)
                inter4 = m4off.intersection(obsp)


                b = [inter1,inter2,inter3,inter4]
                for items in b:
                    if items.geom_type == 'Polygon' and len(items.exterior.coords) >0:
                        self.observobst.append(items)
                        #else:
                        #    for obst_id in range(len(self.observobst)):
                        #        combined_polygon = items.union(self.observobst[obst_id])
                        #        if combined_polygon.geom_type == "Polygon":
                        #            self.observobst[obst_id] = combined_polygon
                        #        else:
                        #            self.observobst.append(items)
              
        cordlist = [list(i.exterior.coords) for i in self.observobst]

        self.collided = 0
        collided,selfcollided = collision(m1,m2,m3,m4,self.obstacle_polygon,self.boundaries,all_pos)

        
        
            

        area = m1.intersection(self.area_polygon).area + m2.intersection(self.area_polygon).area + m3.intersection(self.area_polygon).area+ m4.intersection(self.area_polygon).area
        goal_reward = 0
        if area == 0:
            goal_reward = 10000
            self.done = True
            
        rob_area = m1.area + m2.area + m3.area + m4.area
        offset = rob_area - area

        if self.environment == True:
            if self.pos[0] < 85 or self.pos[1] > 140 and round(m1.intersection(self.area_polygon).area,5) == round(m1.area,5):
                collided = True
        else:
            if self.pos[0] > 140 or self.pos[1] < 85 and round(m1.intersection(self.area_polygon).area,5) == round(m1.area,5) :
                print("yeah")
                collided = True  

        screen.fill(bg_color)
        pygame.draw.polygon(screen, area_color,self.area)

        for o in cordlist:
            pygame.draw.polygon(screen, color, o)




        curpos = self.pos.copy()
        curang = self.ang.copy()
        cur = [headx,heady,ang1,ang2,ang3,ang4]
        if collided:
            self.collided = 1
            self.reward = -2
            self.pos = self.prev_pos.copy()
            self.ang = self.prev_ang.copy()
            
        elif selfcollided:
            self.reward = -2
            self.pos = self.prev_pos.copy()
            self.ang = self.prev_ang.copy()
        else:
            self.prev_pos = self.pos.copy()
            self.prev_ang = self.ang.copy()
            
            if self.preoffset < offset:
                self.preoffset = offset
                self.reward = (rob_area - area) + goal_reward

            elif self.preoffset >= offset:
                self.reward = -0.5
        #elif round(rob_area - area,5) == 0:
        #    self.reward = 0
        #else:
        #    self.reward = rob_area - area + goal_reward + (360 - ang2dif + 360 - ang3dif + 360 - ang4dif)/100
        
        
             
        
        
        self.cumulative += self.reward
        self.prev_actions.append([self.pos[0],self.pos[1],self.ang[0],self.ang[1],self.ang[2],self.ang[3]])
        self.prev_actions.remove(self.prev_actions[0])

        print(curpos,curang,self.reward,self.cumulative,self.count)
        #if self.prev_actions.count(cur) > 2:
        #    self.reward = -0.2
        
        posi = computeOrigins(self.pos,self.ang, self.linkLen)
        
        all_pos = module_pos(posi[0:4], self.ang,self.linkLen2)
        pygame.draw.polygon(screen, rob_color, all_pos[0])
        pygame.draw.polygon(screen, rob_color, all_pos[1])
        pygame.draw.polygon(screen, rob_color, all_pos[2])
        pygame.draw.polygon(screen, rob_color, all_pos[3])
        

        val = pygame.surfarray.array3d(screen)

        val1 = (val / 128).astype(np.uint8)

        single_value_data = (val1[:,:,0]) + (val1[:,:,1]) + (val1[:,:,2])
        #with open('output.csv', 'w', newline='') as csvfile:
        #    writer = csv.writer(csvfile)
        ##    writer.writerow(['R', 'G', 'B'])  # Write header row
        #    writer.writerows(single_value_data)

        m1off = m1off.difference(m1)
        m2off = m2off.difference(m2)
        m3off = m3off.difference(m3)
        m4off = m4off.difference(m4)
        m1off = list(m1off.exterior.coords)
        m2off = list(m2off.exterior.coords)
        m3off = list(m3off.exterior.coords)
        m4off = list(m4off.exterior.coords)

        
        pygame.draw.polygon(screen, offsetcolor, m1off,1)
        pygame.draw.polygon(screen, offsetcolor, m2off,1)
        pygame.draw.polygon(screen, offsetcolor, m3off,1)
        pygame.draw.polygon(screen, offsetcolor, m4off,1)

        pygame.draw.polygon(screen, rob_color, all_pos[0])
        pygame.draw.polygon(screen, rob_color, all_pos[1])
        pygame.draw.polygon(screen, rob_color, all_pos[2])
        pygame.draw.polygon(screen, rob_color, all_pos[3])
        #df.to_excel('output.xlsx', index=False)
        """if self.environment == True:
            BLACK = (0, 0, 0)
            lineStart = (0,self.boundaries[1])
            lineStop = (self.boundaries[2],self.boundaries[1])
            pygame.draw.line(screen, BLACK, lineStart, lineStop, 2)

            lineStart = (0,self.boundaries[3])
            lineStop = (self.boundaries[2],self.boundaries[3])
            pygame.draw.line(screen, BLACK, lineStart, lineStop, 2)

        else:
            BLACK = (0, 0, 0)
            lineStart = (self.boundaries[1],0)
            lineStop = (self.boundaries[1],self.boundaries[2])
            pygame.draw.line(screen, BLACK, lineStart, lineStop, 2)

            lineStart = (self.boundaries[3],0)
            lineStop = (self.boundaries[3],self.boundaries[2])
            pygame.draw.line(screen, BLACK, lineStart, lineStop, 2)"""

        pygame.display.update()
        
        self.observation = {'img' : single_value_data, 'num' :[self.pos[0],self.pos[1],ang1,ang2,ang3,ang4]}    
        
       
        #print(cordlist)
        info = {}
        headx = self.pos[0]
        heady = self.pos[1]
        ang1 = self.ang[0]
        ang2 = self.ang[1]
        ang3 = self.ang[2]
        ang4 = self.ang[3]

        
        return self.observation, self.reward, self.done,info

    def reset(self):
        
        
        self.done = False
        
        if self.z == -5:
            self.incre = 1
        elif self.z == 5:
            self.incre = -1
        
        
        angs = [[270,0,-135,-135],[180,0,-135,-135],[90,0,-135,-135]]
    
        
        self.environment = not self.environment

        if self.environment == True:
            obslist = [40,100,230]
            gap = 40


            boundlist = [(0,0,320,160),(0,70,320,220),(0,150,320,300)]
            extraobs= [[[(24,100),(35,100),(35,120),(24,120)]],[[(24,120),(35,120),(35,140),(24,140)]],[[(24,140),(35,140),(35,160),(24,160)]],[[]]]
            y  = obslist[1]
            obs = [[(0,0),(400,0),(400,24),(0,24)],[(400-24,24),(400,24),(400,300),(400-24,300)],[(400,300-24),(400,300),(0,300),(0,300-24)],[(0,0),(24,0),(24,300),(0,300)]]

            obs1 = [[(96,0),(136,0),(136,y),(96,y)],[(96,y+gap),(136,y+gap),(136,300),(96,300)]]

            obs2 = extraobs[self.counter]
            
            self.obs = obs+obs1
            print(self.obs)

            self.area = [(24,24),(96,24),(96,300-24),(24,300-24)] 


            self.pos = [90,y+30]

            self.ang = angs[1]
            self.boundaries = boundlist[1]
            
        
        else:

            obslist = [40,100,230]
            gap = 40
            boundlist = [(0,0,320,160),(50,0,150,320),(0,150,320,300)]
        
            extraobs= [[[(100,24),(100,35),(120,35),(120,24)]],[[(120,24),(120,35),(140,35),(140,24)]],[[(140,24),(140,35),(160,35),(160,24)]],[[]]]
            y  = obslist[1]
            obs = [[(0,0),(400,0),(400,24),(0,24)],[(400-24,24),(400,24),(400,300),(400-24,300)],[(400,300-24),(400,300),(0,300),(0,300-24)],[(0,0),(24,0),(24,300),(0,300)]]

            obs1 = [[(0,96),(0,136),(y,136),(y,96)],[(y+gap,96),(y+gap,136),(300,136),(300,96)]]

            obs2 = extraobs[self.counter]
            
            self.obs = obs+obs1
            print(self.obs)

            self.area = [(24,24),(24,96),(400-24,96),(400-24,24)] 


            self.pos = [y+gap/2,90]

            self.ang = angs[0]
            self.boundaries = boundlist[1]

           


        self.counter += 1
        if self.counter>2:
            self.counter = 0
        #    self.environment = not self.environment

        self.count+=1
        
       
        self.obstacle_polygon = obst_poly(self.obs)
        self.area_polygon = Polygon(self.area)
       

        self.collided = 0

        headx = self.pos[0]
        heady = self.pos[1]
        ang1 = self.ang[0]
        ang2 = self.ang[1]
        ang3 = self.ang[2]
        ang4 = self.ang[3]

       
        self.prev_actions = [deque(maxlen = 5)]

        posi = computeOrigins(self.pos,self.ang, self.linkLen)
        
        all_pos = module_pos(posi[0:4], self.ang,self.linkLen2)

        m1 = Polygon(all_pos[0])
        m2 = Polygon(all_pos[1])
        m3 = Polygon(all_pos[2])
        m4 = Polygon(all_pos[3])
        
        rob_area = m1.area + m2.area + m3.area + m4.area
        area = m1.intersection(self.area_polygon).area + m2.intersection(self.area_polygon).area + m3.intersection(self.area_polygon).area+ m4.intersection(self.area_polygon).area
        
        offset = rob_area - area

        screen.fill(bg_color)
        pygame.draw.polygon(screen, area_color,self.area)

        for o in self.obs:
            pygame.draw.polygon(screen, color, o)
        pygame.draw.polygon(screen, rob_color, all_pos[0])
        pygame.draw.polygon(screen, rob_color, all_pos[1])
        pygame.draw.polygon(screen, rob_color, all_pos[2])
        pygame.draw.polygon(screen, rob_color, all_pos[3])

        val = pygame.surfarray.array3d(screen)
        val1 = (val / 128).astype(np.uint8)

        single_value_data = (val1[:,:,0]) + (val1[:,:,1]) + (val1[:,:,2])
        

      

        for _ in range(5):
           self.prev_actions.append([-1,-1,-1,-1,-1,-1])

        self.prev_pos =self.pos.copy()
        self.prev_ang = self.ang.copy()
        self.observation = {'img' : single_value_data, 'num' :[self.pos[0],self.pos[1],ang1,ang2,ang3,ang4]}
        self.reward = 0
        self.rep = 0
        self.max_reward = 0
        self.z += self.incre
        self.cumulative = 0
        self.preoffset = 0
        return self.observation


