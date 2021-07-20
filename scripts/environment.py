import os
import sys
import glob

# Find CARLA module
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import cv2
import time
import math
import carla
import random
import numpy as np

SEMANTIC_IMG_HEIGHT = 75
SEMANTIC_IMG_WIDTH = 200
RGB_IMG_HEIGHT = 600
RGB_IMG_WIDTH = 800
FOCAL_LENGTH = 36.66184120297396
SECONDS_PER_EPISODE = 30

class Env:
    def __init__(self): 
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model3 = self.blueprint_library.filter('model3')[0]
        self.cybertruck = self.blueprint_library.filter('cybertruck')[0]

    def reset(self):
        self.ego = None; self.lead = None; self.rgb_image = None; self.semantic_image = None
        self.prev_kmph = 0; self.actor_list = []; self.skip_episode = False

        while self.ego == None:
            lead_loc, ego_loc = self.set_behind()
            self.spawn_vehicle(lead_loc, ego_loc)
        self.attach_sensors()

        # To reduce latency in the vehicle responding to a command
        self.lead.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        self.ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        while self.rgb_image is None or self.semantic_image is None:
            time.sleep(0.01)
        
        self.episode_start = time.time()
        lead_speed = random.uniform(0.0, 0.50)
        self.ego_speed = random.uniform(0.65, 0.90)
        self.lead.apply_control(carla.VehicleControl(throttle=lead_speed))
        self.ego.apply_control(carla.VehicleControl(throttle=self.ego_speed))
        
        if not self.get_distance(self.semantic_image): # no vehicle in front
            self.destroy_actor(); self.skip_episode = True

        return self.rgb_image, self.ego_speed, self.skip_episode
        
    # Spawn vehicle if proposed location of ego and lead are not equivalent
    def spawn_vehicle(self, lead_loc, ego_loc): 
        if lead_loc.location.x != ego_loc.location.x \
            or lead_loc.location.y != ego_loc.location.y:
            try:
                self.lead = self.world.spawn_actor(self.cybertruck, lead_loc)
                self.actor_list.append(self.lead)
                self.ego = self.world.spawn_actor(self.model3, ego_loc)
                self.actor_list.append(self.ego)   
            except:
                self.destroy_actor()

    # Position ego behind lead
    def set_behind(self): 
        lead_location = random.choice(self.world.get_map().get_spawn_points())
        ego_location = carla.Transform(lead_location.location, lead_location.rotation)
        
        if lead_location.rotation.yaw > 80 and lead_location.rotation.yaw < 100:
            ego_location.location.y -= random.uniform(20, 30)
        elif lead_location.rotation.yaw < -80 and lead_location.rotation.yaw > -100:
            ego_location.location.y += random.uniform(20, 30)
        elif lead_location.rotation.yaw > 170 and lead_location.rotation.yaw < 190:
            ego_location.location.x -= random.uniform(20, 30)
        elif lead_location.rotation.yaw < -170 and lead_location.rotation.yaw > -190:
            ego_location.location.x += random.uniform(20, 30)

        return lead_location, ego_location

    def attach_sensors(self):
        semantic_camera = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        semantic_camera.set_attribute('image_size_x', f'{SEMANTIC_IMG_WIDTH}')
        semantic_camera.set_attribute('image_size_y', f'{SEMANTIC_IMG_HEIGHT}')
        rgb_camera = self.blueprint_library.find('sensor.camera.rgb')
        rgb_camera.set_attribute('image_size_x', f'{RGB_IMG_WIDTH}')
        rgb_camera.set_attribute('image_size_y', f'{RGB_IMG_HEIGHT}')
   
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.semantic_camera = self.world.spawn_actor(semantic_camera, transform, attach_to=self.ego)
        self.semantic_camera.listen(lambda image: self.preprocess_semantic(image))
        self.rgb_camera = self.world.spawn_actor(rgb_camera, transform, attach_to=self.ego)
        self.rgb_camera.listen(lambda image: self.preprocess_rgb(image))
        self.actor_list.append(self.semantic_camera)
        self.actor_list.append(self.rgb_camera)

    def preprocess_semantic(self, image):
        image_data = np.array(image.raw_data)
        image_data = image_data.reshape((SEMANTIC_IMG_HEIGHT, SEMANTIC_IMG_WIDTH, 4))
        image_data = image_data[:,:,:3]
        self.semantic_image = image_data

    def preprocess_rgb(self, image):
        image_data = np.array(image.raw_data)
        image_data = image_data.reshape((RGB_IMG_HEIGHT, RGB_IMG_WIDTH, 4))       
        image_data = image_data[300:600,0:800,:]
        hmin = 0; hmax = 255; smin = 0; smax = 80; vmin = 180; vmax = 255
        bgrImage = cv2.cvtColor(image_data, cv2.COLOR_BGRA2BGR)
        yuvImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2YUV)
        yuvImage[:,:,0] = cv2.equalizeHist(yuvImage[:,:,0])
        yuvImage[:,0,:] = cv2.equalizeHist(yuvImage[:,0,:])
        yuvImage[0,:,:] = cv2.equalizeHist(yuvImage[0,:,:])

        normalized = cv2.cvtColor(yuvImage, cv2.COLOR_YUV2RGB)
        hsvImage = cv2.cvtColor(normalized, cv2.COLOR_RGB2HSV)
        lower = (hmin, smin, vmin)
        upper = (hmax, smax, vmax)
        filter = cv2.inRange(hsvImage, lower, upper)

        edgeImage = cv2.Canny(filter, 100, 200)
        img = cv2.resize(edgeImage, (200, 75))
        self.rgb_image = img
    
    def step(self, action, distance): 
        if distance < 3.5:
            if action == 0:
                pass
            elif action == 1:
                self.ego.apply_control(carla.VehicleControl(brake=0.33))
            elif action == 2:
                self.ego.apply_control(carla.VehicleControl(brake=0.67))
            elif action == 3:
                self.ego.apply_control(carla.VehicleControl(brake=1))

        velocity = self.ego.get_velocity()
        kmph = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)   
        deceleration = kmph - self.prev_kmph 
        alpha = 0.001; beta = 0.1; mu = 0.01; nu = 100; bump = False; done = False

        if distance < 1.25:
            bump = True; done = True

        """
        Agent reward function consists of two parts:
            1. LHS has small punishment for unneccesary braking:
                - Braking while being far from lead car
                - Slamming on brakes, not gradually slowing down
            2. RHS has large punishment for crashing into lead car:
                - Speed of ego car before crash
            
            alpha, beta, mu, and nu are hyperparameters. By tuning them, you teach the agent to care
            more about certain things

        """
        reward = -(alpha * (distance)**2 + beta) * deceleration - (mu * kmph**2 + nu) * bump     
        if reward > 0:
            reward = 0

        self.prev_kmph = kmph

        # Episode cannot exceed set amount of seconds
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.rgb_image, reward, done, None
    
    def get_focal_length(self, semantic_image):
        width = self.lead.bounding_box.extent.x * 2
        distance = 10; percieved_pixel_count = -1; vehicle_pixel_count = 0
        for layer in semantic_image:
            for pixel in layer:
                if pixel[2] == 10:
                    vehicle_pixel_count += 1
            if percieved_pixel_count < vehicle_pixel_count:
                percieved_pixel_count = vehicle_pixel_count
            vehicle_pixel_count = 0

        return (percieved_pixel_count * distance) / width

    def get_distance(self, semantic_image):
        percieved_pixel_count = 0; vehicle_pixel_count = 0
        for layer in semantic_image:
            for pixel in layer:
                if pixel[2] == 10:
                    vehicle_pixel_count += 1
            if percieved_pixel_count < vehicle_pixel_count:
                percieved_pixel_count = vehicle_pixel_count
            vehicle_pixel_count = 0

        try:
            width = self.lead.bounding_box.extent.x * 2
            distance = (width * FOCAL_LENGTH) / percieved_pixel_count
        except:
            distance = None
            self.destroy_actor()
        return distance        

    def destroy_actor(self):
        for actor in self.actor_list:
            if actor.is_alive:
                if actor.__class__ != carla.libcarla.Vehicle:
                    actor.stop()
                actor.destroy()

        

