import time
import os

import numpy as np
import gym
from gym import spaces

from aigar.envs.model.bot import Bot
from aigar.envs.model.field import Field
from aigar.envs.model.parameters import *
from aigar.envs.model.player import Player
from aigar.envs.model.rgbGenerator import RGBGenerator

# The aigar class is the main wrapper for the game engine.
# It contains the field and the players.
# It links the actions of the players to consequences in the field and updates information.

class AigarEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, rgb=False, num_greedy=0, split=False, eject=False):
        super(AigarEnv, self).__init__()
        self.rgb = rgb
        self.enable_split = split
        self.enable_eject = eject
        if num_greedy == 0:
            self.use_enemy_grid = False
        else:
            self.use_enemy_grid = True
        
        self.listeners = []
        self.virusEnabled = VIRUS_SPAWN
        self.resetLimit = RESET_LIMIT
        self.startTime = None

        self.players = []
        self.bots = []
        self.humans = []
        self.playerSpectator = None
        self.spectatedPlayer = None
        self.field = Field(self.virusEnabled)
        self.screenWidth = None
        self.screenHeight = None
        self.counter = 0
        self.timings = []
        self.rewards = []
        self.dataFiles = {}

        self.viewer= None
        # Set up model:
        self.gym_bot = self.createBot("Gym")
        for _ in range(num_greedy):
            self.createBot("Greedy", None, None)
        self.initialize()
        # Set up spaces:
        self.action_space = self.create_action_space()
        self.num_actions = len(self.action_space.low)
        self.observation_space = self.create_observation_space()
    
    # Start Interface for gym env:
    def step(self, action):
        if len(action) != self.num_actions:
            raise TypeError("The number of dimensions of the action does not match the action space!")
        self.update(action=action)
        obs, reward, done = self.getStepData()
        return obs, reward, done, {}
        
    def reset(self):
        self.field.reset()
        self.resetBots()
        self.counter = 0
        state = self.get_state()
        return state
        
    def render(self, mode="human", close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None      # If we don't None out this reference pyglet becomes unhappy
            return
        #try:
        #if 'human' == mode and self.no_render:

        #    return
        #state = self.game.get_state()
        #img = state.image_buffer
        img = self.gym_bot.rgbGenerator.get_cnn_inputRGB(self.gym_bot.player)
        
        if img is None:
            img = np.zeros(shape=self.observation_space.shape, dtype=np.uint8)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def get_state(self):
        if self.rgb:
            state = self.gym_bot.rgbGenerator.get_cnn_inputRGB(self.gym_bot.player)
        else:
            state = self.gym_bot.getGridStateRepresentation()
        return state
        
    def getStepData(self):
        state = get_state()
        reward = self.gym_bot.getReward()
        alive = self.gym_bot.player.getIsAlive()
        done = not alive
        return state, reward, done
                
    def update(self, action=None):
        self.counter += 1
        # Get the decisions of the bots. Update the field accordingly.
        self.takeBotActions(action)
        self.field.update()
        
    def create_action_space(self):
        # attack or move, move_degree, move_distance
        action_low = [0.0, 0.0]
        action_high = [1.0, 1.0]
        if self.enable_split:
            action_low += [0.0]
            action_high += [1.0]
        if self.enable_eject:
            action_low += [0.0]
            action_high += [1.0]
            
        return spaces.Box(np.array(action_low, dtype=np.float32),
                          np.array(action_high, dtype=np.float32), dtype=np.float32)

    def rgb_space(self):
        img = self.gym_bot.rgbGenerator.get_cnn_inputRGB(self.gym_bot.player)
        obs_low = np.zeros_like(img)
        obs_high = np.ones_like(img) * 255
        return spaces.Box(np.array(obs_low, dtype=np.uint8),
                          np.array(obs_high, dtype=np.uint8), dtype=np.uint8)
        
    def grid_space(self):
        obs_shape = self.gym_bot.getObsSize()
        obs_low = np.zeros(obs_shape)
        obs_high = np.ones(obs_shape) * math.inf
        return spaces.Box(np.array(obs_low, dtype=np.float32),
                          np.array(obs_high, dtype=np.float32), dtype=np.float32)
    
    def create_observation_space(self):
        if self.rgb:
            return self.rgb_space()
        else:
            return self.grid_space()
        
    def initParameters(self, parameters):
        self.parameters = parameters
        self.virusEnabled = parameters.VIRUS_SPAWN
        self.resetLimit = parameters.RESET_LIMIT
        # self.pointAveraging = parameters.EXPORT_POINT_AVERAGING
        self.field = Field(self.virusEnabled)

    def modifySettings(self, reset_time):
        self.resetLimit = reset_time

    def initialize(self):
        self.field.initialize()
        self.resetBots()

    def takeBotActions(self, action):
        for bot in self.bots:
            bot.makeMove(action)

    def resetBots(self):
        for bot in self.bots:
            bot.reset()

    def printBotMasses(self):
        for bot in self.bots:
            mass = bot.getPlayer().getTotalMass()
            print("Mass of ", bot.getPlayer(), ": ", round(mass, 1) if mass is not None else "Dead")

    def createPlayer(self, name):
        newPlayer = Player(name)
        self.addPlayer(newPlayer)
        return newPlayer

    def createBot(self, botType, learningAlg = None, parameters = None):
        name = botType + str(len(self.bots))
        newPlayer = self.createPlayer(name)
        rgbGenerator = None
        if botType == "Gym":
            rgbGenerator = RGBGenerator(self.field, parameters)
        bot = Bot(newPlayer, self.field, botType, learningAlg, parameters, rgbGenerator,
                  use_enemy_grid=self.use_enemy_grid)
        self.addBot(bot)
        return bot

    def createHuman(self, name):
        newPlayer = self.createPlayer(name)
        self.addHuman(newPlayer)

    def addPlayer(self, player):
        self.players.append(player)
        self.field.addPlayer(player)

    def addBot(self, bot):
        self.bots.append(bot)
        player = bot.getPlayer()
        if player not in self.players:
            self.addPlayer(player)

    def addHuman(self, human):
        self.humans.append(human)

    def addPlayerSpectator(self):
        self.playerSpectator = True
        self.setSpectatedPlayer(self.players[0])

    def setPath(self, path):
        self.path = path

    def setSpectatedPlayer(self, player):
        self.spectatedPlayer = player

    def setViewEnabled(self, boolean):
        self.viewEnabled = boolean

    def setScreenSize(self, width, height):
        self.screenWidth = width
        self.screenHeight = height

    # Checks:
    def hasHuman(self):
        return bool(self.humans)

    def hasPlayerSpectator(self):
        return self.playerSpectator is not None

    # Getters:
    def getNNBot(self):
        for bot in self.bots:
            if bot.getType() == "NN":
                return bot

    def getNNBots(self):
        return [bot for bot in self.bots if bot.getType() == "NN"]

    def getTopTenPlayers(self):
        players = self.getPlayers()[:]
        players.sort(key=lambda p: p.getTotalMass(), reverse=True)
        return players[0:10]


    def getHumans(self):
        return self.humans

    def getFovPos(self, humanNr):
        if self.hasHuman():
            fovPos = np.array(self.humans[humanNr].getFovPos())
        elif self.hasPlayerSpectator():
            fovPos = np.array(self.spectatedPlayer.getFovPos())
        else:
            fovPos = np.array([self.field.getWidth() / 2, self.field.getHeight() / 2])
        return fovPos

    def getFovSize(self, humanNr):
        if self.hasHuman():
            fovSize = self.humans[humanNr].getFovSize()
        elif self.hasPlayerSpectator():
            fovSize = self.spectatedPlayer.getFovSize()
        else:
            fovSize = self.field.getWidth()
        return fovSize

    def getField(self):
        return self.field

    def getPellets(self):
        return self.field.getPellets()

    def getViruses(self):
        return self.field.getViruses()

    def getPlayers(self):
        return self.players

    def getBots(self):
        return self.bots

    def getPlayerCells(self):
        return self.field.getPlayerCells()

    def getSpectatedPlayer(self):
        if self.hasHuman():
            return self.humans
        if self.hasPlayerSpectator():
            return self.spectatedPlayer
        return None

    def getParameters(self):
        return self.parameters

    def getVirusEnabled(self):
        return self.virusEnabled

    # MVC related method
    def set_GUI(self, value):
        self.guiEnabled = value

    def register_listener(self, listener):
        self.listeners.append(listener)

    def notify(self):
        for listener in self.listeners:
            listener()
