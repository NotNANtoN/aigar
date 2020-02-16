import time
import datetime
import matplotlib
matplotlib.use('Pdf')
import matplotlib.pyplot as plt
import numpy as np

from .model.bot import Bot
from .model.field import Field
from .model.parameters import *
from .model.player import Player
from .model.rgbGenerator import RGBGenerator

import linecache
import os
import tracemalloc
import pickle as pkl
import gym
from gym import spaces


# The aigar class is the main wrapper for the game engine.
# It contains the field and the players.
# It links the actions of the players to consequences in the field and updates information.

class AigarGreedyEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(AigarGreedyEnv, self).__init__()
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
        self.createBot("Gym")
        self.createBot("Greedy")
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()
        #self.createBot("Greedy", None, None)
        self.initialize()
    
    # Start Interface for gym env:
    def step(self, action):
        self.update(action=action)
        
        obs, reward, done = self.getStepData()
        
        return obs, reward, done, {}
        
    
    def reset(self):
        self.field.reset()
        self.resetBots()
        self.counter = 0
        
    
    def render(self, mode="human", close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None      # If we don't None out this reference pyglet becomes unhappy
            return
        #try:
        #if 'human' == mode and self.no_render:
        #    print("Close because no render")
        #    return
        #state = self.game.get_state()
        #img = state.image_buffer
        img = self.gym_bot.rgbGenerator.get_cnn_inputRGB(bot.player)
        
        if img is None:
            img = np.zeros(shape=self.observation_space.shape, dtype=np.uint8)
        if mode == 'rgb_array':
            return img
        elif mode is 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
        #except Exception:
        #    print(Exception) # Game has been closed
        
    def _action_space(self):
        # attack or move, move_degree, move_distance
        action_low = [0.0, 0.0, 0.0, 0.0]
        action_high = [1.0, 1.0, 1.0, 1.0]
        return spaces.Box(np.array(action_low, dtype=np.float32), np.array(action_high, dtype=np.float32))

    def _observation_space(self):
        # hit points, cooldown, ground range, is enemy, degree, distance (myself)
        # hit points, cooldown, ground range, is enemy (enemy)
        obs_size = self.getObsSpaceSize()
        obs_low = [0.0 for _ in range(obs_size)]
        obs_high = [math.inf for _ in range(obs_size)]
        
        return spaces.Box(np.array(obs_low, dtype=np.float32), np.array(obs_high, dtype=np.float32))
        
    # End Interface for gym env

    def initParameters(self, parameters):
        self.parameters = parameters
        self.virusEnabled = parameters.VIRUS_SPAWN
        self.resetLimit = parameters.RESET_LIMIT
        # self.pointAveraging = parameters.EXPORT_POINT_AVERAGING
        self.field = Field(self.virusEnabled)


    def modifySettings(self, reset_time):
        self.resetLimit = reset_time

    def initialize(self):
        if __debug__:
            print("Initializing model...")
        self.field.initialize()
        self.resetBots()

    def getStepData(self):
        return self.gym_bot.getGridStateRepresentation(), self.gym_bot.getReward(), self.gym_bot.player.getIsAlive()
    
    def getObsSpaceSize(self):
        return self.gym_bot.getObsSize()
                
    def update(self, action=None):
        self.counter += 1

        #timeStart = time.time()
        # Get the decisions of the bots. Update the field accordingly.
        self.takeBotActions(action)
        self.field.update()
        # Update view if view is enabled
        #if self.guiEnabled and self.viewEnabled:
        #    self.notify()
        # Slow down game to match FPS (disabled in gym mode)
        #if self.humans:
        #    time.sleep(max( (1/FPS) - (time.time() - timeStart),0))

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
        bot = Bot(newPlayer, self.field, botType, learningAlg, parameters, rgbGenerator)
        self.addBot(bot)

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
