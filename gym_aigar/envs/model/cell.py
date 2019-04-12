import numpy
import math
from .parameters import *



class Cell(object):
    _cellId = 0

    @property
    def cellId(self):
        return type(self)._cellId

    @cellId.setter
    def cellId(self, val):
        type(self)._cellId = val

    def __repr__(self):
        return self.name + " id: " + str(self.id) + " -M:" + str(int(self.mass)) + " Pos:" + str(int(self.x)) + "," + str(int(self.y))

    def __init__(self, x, y, mass, player):
        self.player = player
        self.mass = None
        self.radius = None
        self.setMass(mass)
        self.x = x
        self.y = y
        self.pos = [x,y]
        if self.player is None:
            self.name = ""
            self.color = (numpy.random.randint(50, 200), numpy.random.randint(50, 200), numpy.random.randint(50, 200))
            self.id = -1
        else:
            self.name = player.getName()
            self.color = self.player.getColor()
            self.id = self.cellId
            self.cellId += 1
        self.velocity = [0, 0]
        self.splitVelocity = [0, 0]
        self.splitVelocityCounter = 0
        self.splitVelocityCounterMax = 15
        self.mergeTime = 0
        self.blobToBeEjected = None
        self.ejecterCell = None # Used in case of blobs to determine which player ejected this blob
        self.alive = True

    def setMoveDirection(self, commandPoint):
        xDiff = commandPoint[0] - self.x
        yDiff = commandPoint[1] - self.y
        # If cursor is within cell, reduce speed based on distance from cell center (as a percentage)
        hypotenuseSquared = xDiff * xDiff + yDiff * yDiff
        radiusSquared = self.radius * self.radius
        speedModifier = min(hypotenuseSquared, radiusSquared) / radiusSquared
        # Check polar coordinate of cursor from cell center
        angle = math.atan2(yDiff, xDiff)
        self.velocity[0] =  self.getReducedSpeed() * speedModifier * math.cos(angle)
        self.velocity[1] =  self.getReducedSpeed() * speedModifier * math.sin(angle)

    @staticmethod
    def squareDist(pos1, pos2):
        return (pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + (pos1[1] - pos2[1]) * (pos1[1] - pos2[1])

    @staticmethod
    def updateDirection(x, v, maxX):
        return min(maxX, max(0, x + v))

    def calculateAngle(self, point):
        xDiff = point[0] - self.x
        yDiff = point[1] - self.y
        return math.atan2(yDiff, xDiff)

    def split(self, commandPoint, fieldWidth, fieldHeight):
        cellPos = self.getPos()
        newCell = Cell(cellPos[0], cellPos[1], self.mass / 2, self.player)
        angle = newCell.calculateAngle(commandPoint)

        xPoint = math.cos(angle) * newCell.getRadius() * 4.5 + cellPos[0]
        yPoint = math.sin(angle) * newCell.getRadius() * 4.5 + cellPos[1]
        movePoint = (xPoint, yPoint)
        #newCell.setMoveDirection(movePoint)
        newCell.addMomentum(movePoint, fieldWidth, fieldHeight, self)
        newCell.resetMergeTime(1)
        self.setMass(self.mass / 2)
        #self.resetMergeTime(1)
        return newCell

    def prepareEject(self):
        self.blobToBeEjected = True

    def eject(self):
        #blobSpawnPos can be None if commandPoint is in center of cell, in which case nothing is ejected
        self.mass -= EJECTEDBLOB_BASE_MASS
        self.blobToBeEjected = False
        return self.getPos()

    def addMomentum(self, commandPoint, fieldWidth, fieldHeight, originalCell):
        checkedX = max(0, min(fieldWidth, commandPoint[0]))
        checkedY = max(0, min(fieldHeight, commandPoint[1]))
        checkedPoint = (checkedX, checkedY)
        angle = self.calculateAngle(checkedPoint)
        speed = 2 + originalCell.getRadius() * 0.05
        self.splitVelocity = [math.cos(angle) * speed, math.sin(angle) * speed]
        self.splitVelocityCounter = self.splitVelocityCounterMax

    def updateMomentum(self):
        if self.splitVelocityCounter == -1:
            return
        elif self.splitVelocityCounter > 0:
            self.splitVelocityCounter -= 1
            counterRatio = self.splitVelocityCounter / self.splitVelocityCounterMax
            if counterRatio < 0.1:
                self.splitVelocity[0] *= (1 - counterRatio)
                self.splitVelocity[1] *= (1 - counterRatio)
        else:
            self.splitVelocity = [0,0]
            self.splitVelocityCounter = -1

    # Increases the mass of the cell by value and updates the radius accordingly
    def grow(self, foodMass):
        newMass = min(MAX_MASS_SINGLE_CELL, self.mass + foodMass)
        self.setMass(newMass)

    def decayMass(self):
        if self.mass >= 4:
            newMass = self.mass * CELL_MASS_DECAY_RATE
            self.setMass(newMass)

    def updateMerge(self):
        if self.mergeTime > 0:
            self.mergeTime -= 1

    def updatePos(self, maxX, maxY):
        xSpeed = self.velocity[0] + self.splitVelocity[0]
        ySpeed = self.velocity[1] + self.splitVelocity[1]
        self.x = self.updateDirection(self.x, xSpeed, maxX)
        self.y = self.updateDirection(self.y, ySpeed, maxY)
        self.pos = [self.x, self.y]
        if self.splitVelocityCounter and self.x == maxX or self.x == 0:
            self.splitVelocity[0] *= -1
        if self.splitVelocityCounter and self.y == maxY or self.y == 0:
            self.splitVelocity[1] *= -1

    def overlap(self, cell):
        if self.getMass() > cell.getMass():
            biggerCell = self
            smallerCell = cell
        else:
            biggerCell = cell
            smallerCell = self
        if biggerCell.squaredDistance(smallerCell) * 1.1 < biggerCell.getRadius() * biggerCell.getRadius():
            return True
        return False

    def resetMergeTime(self, factor):
        self.mergeTime = factor * (BASE_MERGE_TIME + self.mass * MERGE_TIME_MASS_FACTOR) * FPS / 2 / GAME_SPEED

    # Returns the squared distance from the self cell to another cell
    def squaredDistance(self, cell):
        pos2 = cell.getPos()
        return (self.x - pos2[0]) * (self.x - pos2[0]) + (self.y - pos2[1]) * (self.y - pos2[1])

    # Checks:
    def canEat(self, cell):
        return self.mass > 1.25 * cell.getMass()

    def isAlive(self):
        return self.alive == True

    def isInFov(self, fovPos, fovSize):
        halvedFovDims = fovSize / 2
        xMin = fovPos[0] - halvedFovDims
        xMax = fovPos[0] + halvedFovDims
        yMin = fovPos[1] - halvedFovDims
        yMax = fovPos[1] + halvedFovDims
        if self.x + self.radius < xMin or self.x - self.radius > xMax or self.y + self.radius < yMin or self.y - self.radius > yMax:
            return False
        return True

    def justEjected(self):
        return self.splitVelocityCounter > 0

    def canSplit(self):
        return self.mass > 36

    def canEject(self):
        return self.mass >= 35

    def canMerge(self):
        return self.mergeTime <= 0

    # Setters:
    def setColor(self, color):
        self.color = color

    def setName(self, name):
        self.name = name

    def setAlive(self, val):
        self.alive = val

    def setPos(self, pos):
        self.pos = pos
        self.x = pos[0]
        self.y = pos[1]

    def setRadius(self, val):
        self.radius = val
        self.mass = self.radius * self.radius * numpy.pi

    def setMass(self, val):
        self.mass = val
        self.radius = math.sqrt(val / numpy.pi) if val > 0 else 0

    def setBlobToBeEjected(self, val):
        self.blobToBeEjected = False

    def setEjecterCell(self, cell):
        self.ejecterCell = cell
        self.color = cell.getColor()

    # Getters:
    def getMergeTime(self):
        return self.mergeTime

    def getSplitVelocityCounter(self):
        return self.splitVelocityCounter

    def getPlayer(self):
        return self.player

    def getName(self):
        return self.name

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getPos(self):
        return self.pos

    def getColor(self):
        return self.color

    def getRadius(self):
        return self.radius

    def getMass(self):
        return self.mass

    def getReducedSpeed(self):
        #return CELL_MOVE_SPEED * math.pow(self.mass, -0.439)
        return CELL_MOVE_SPEED * math.pow(self.mass, -0.35)

    def getVelocity(self):
        return self.velocity

    def getSplitVelocity(self):
        return self.splitVelocity

    def getBlobToBeEjected(self):
        return self.blobToBeEjected

    def getEjecterCell(self):
        return self.ejecterCell
