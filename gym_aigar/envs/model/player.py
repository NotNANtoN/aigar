from random import randint
from numpy import sum

class Player(object):
    """docstring for Player"""
    stepsUntilRespawn = 1

    def __repr__(self):
        return self.name

    def __init__(self, name):
        self.color = None
        self.randomizeColor()
        self.name = name
        self.cells = []
        self.canSplit = False
        self.canEject = False
        self.isAlive = True
        # Commands:
        self.commandPoint = [-1, -1]
        self.doSplit = False
        self.doEject = False
        self.fovPos = []
        self.fovSize = None
        self.respawnTime = 0

        self.selected = False
        self.exploring = False

    def update(self, fieldWidth, fieldHeight):
        if self.isAlive:
            self.decayMass()
            self.updateCellProperties()
            self.split(fieldWidth, fieldHeight)
            self.eject()
            self.updateCellsMovement(fieldWidth, fieldHeight)

    def randomizeColor(self):
        self.color = (randint(0, 255), randint(0, 255), randint(0, 255))
        if sum(self.color) > 600:
            self.randomizeColor()

    def decayMass(self):
        for cell in self.cells:
            cell.decayMass()

    def updateCellProperties(self):
        for cell in self.cells:
            cell.updateMomentum()
            cell.updateMerge()
            cell.setMoveDirection(self.commandPoint)

    def split(self, fieldWidth, fieldHeight):
        if not self.doSplit:
            return
        self.cells.sort(key=lambda p: p.getMass(), reverse=True)
        newCells = []
        for cell in self.cells[:]:
            if cell.canSplit() and len(self.cells) + len(newCells) < 16:
                newCell = cell.split(self.commandPoint, fieldWidth, fieldHeight)
                self.addCell(newCell)

    def eject(self):
        if not self.doEject:
            return
        for cell in self.cells:
            if cell.canEject():
                cell.prepareEject()

    def updateCellsMovement(self, fieldWidth, fieldHeight):
        for cell in self.cells:
            cell.updatePos(fieldWidth, fieldHeight)

    def updateRespawnTime(self):
        self.respawnTime -= 1

    # Setters:
    def setExploring(self, val):
        self.exploring = val

    def setSelected(self, val):
        self.selected = val

    def setMoveTowards(self, relativeMousePos):
        self.commandPoint = relativeMousePos

    def addCell(self, cell):
        self.cells.append(cell)

    def addMass(self, value):
        for cell in self.cells:
            mass = cell.getMass()
            cell.setMass(mass + value)

    def removeCell(self, cell):
        cell.setAlive(False)
        self.cells.remove(cell)

    def setCommands(self, x, y, split, eject):
        self.commandPoint = [x, y]
        self.doSplit = split
        self.doEject = eject

    def setSplit(self, val):
        self.doSplit = val

    def setEject(self, val):
        self.doEject = val

    def setDead(self):
        self.isAlive = False
        self.respawnTime = self.stepsUntilRespawn

    def setAlive(self):
        self.isAlive = True
        self.respawnTime = 0

    # Checks:
    def isExploring(self):
        return self.exploring
    # Getters:

    def getSelected(self):
        return self.selected

    def getRespawnTime(self):
        return self.respawnTime

    def getTotalMass(self):
        return sum([cell.getMass() for cell in self.cells]) if self.cells else 0

    def getCells(self):
        return self.cells

    def getMergableCells(self):
        cells = []
        for cell in self.cells:
            if cell.canMerge():
                cells.append(cell)
        return cells

    def getCanSplit(self):
        if len(self.cells) >= 16:
            return False
        for cell in self.cells:
            if cell.canSplit():
                return True
        return False

    def getCanEject(self):
        for cell in self.cells:
            if cell.canEject():
                return True
        return False

    def getFovPos(self):
        if self.isAlive and self.getTotalMass() != 0:
            meanX = sum([cell.getX() * cell.getMass() for cell in self.cells]) / self.getTotalMass()
            meanY = sum([cell.getY() * cell.getMass() for cell in self.cells]) / self.getTotalMass()
            self.fovPos = [meanX, meanY]
        return self.fovPos

    def getFovSize(self):
        if self.isAlive:
            biggestCellRadius = max(self.cells, key=lambda p: p.getRadius()).getRadius()
            self.fovSize = (biggestCellRadius ** 0.475) * (len(self.cells) ** 0.32) * 35
        return self.fovSize

    def getFov(self):
        fovPos = self.getFovPos()
        fovSize = self.getFovSize()
        return fovPos, fovSize

    def getColor(self):
        return self.color

    def getName(self):
        return self.name

    def getIsAlive(self):
        return self.isAlive

    def getCommandPoint(self):
        return self.commandPoint
