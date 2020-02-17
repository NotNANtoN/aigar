import os
import time

import numpy

from .cell import Cell
from .parameters import *
from .spatialHashTable import SpatialHashTable


def adjustCellSize(cell, mass, hashtable):
    hashtable.deleteObject(cell)
    cell.grow(mass)
    hashtable.insertObject(cell)


def randomSize():
    maxRand = 50
    maxPelletSize = 4
    sizeRand = numpy.random.randint(0, maxRand)
    if sizeRand > (maxRand - maxPelletSize):
        return maxRand - sizeRand
    return 1


class Field(object):
    """ The Field class is the main field on which cells of all sizes will move
    Its size depends on how many players are in the game
    It always contains a certain number of viruses and collectibles and regulates their number and spawnings
    """
    def __init__(self, virusEnabled):
        # Set numpy seed to process ID * current time, otherwise all subprocesses have same outcome
        numpy.random.seed(int(time.time())%os.getpid())
        self.size = 0
        self.pellets = []
        self.players = []
        self.blobs = [] # Ejected particles become pellets once momentum is lost
        self.deadPlayers = []
        self.viruses = []
        self.maxCollectibleCount = None
        self.maxVirusCount = None
        self.pelletHashTable = None
        self.blobHashTable = None
        self.playerHashTable = None
        self.virusHashTable = None

        self.virusEnabled = virusEnabled


    def initializePlayer(self, player):
        player.randomizeColor()
        player.cells = []
        x, y = self.getSpawnPos(START_RADIUS)
        newCell = Cell(x, y, START_MASS, player)
        player.addCell(newCell)
        player.setAlive()

    def initialize(self):
        self.size = int(SIZE_INCREASE_PER_PLAYER * math.sqrt(len(self.players)))
        self.pelletHashTable = SpatialHashTable(self.size, HASH_BUCKET_SIZE)
        self.blobHashTable = SpatialHashTable(self.size, HASH_BUCKET_SIZE)
        self.playerHashTable = SpatialHashTable(self.size, HASH_BUCKET_SIZE)
        self.virusHashTable = SpatialHashTable(self.size, HASH_BUCKET_SIZE)
        for player in self.players:
            self.initializePlayer(player)
        self.maxCollectibleCount = self.size * self.size * MAX_COLLECTIBLE_DENSITY
        self.maxVirusCount = self.size * self.size * MAX_VIRUS_DENSITY
        self.spawnStuff()

    def reset(self):
        # Clear field
        self.pellets = []
        self.blobs = []  # Ejected particles become pellets once momentum is lost
        self.deadPlayers = []
        self.viruses = []
        self.pelletHashTable = SpatialHashTable(self.size, HASH_BUCKET_SIZE)
        self.blobHashTable = SpatialHashTable(self.size, HASH_BUCKET_SIZE)
        self.playerHashTable = SpatialHashTable(self.size, HASH_BUCKET_SIZE)
        self.virusHashTable = SpatialHashTable(self.size, HASH_BUCKET_SIZE)

        # Spawn stuff
        for player in self.players:
            self.initializePlayer(player)
        self.spawnStuff()

    def update(self):
        self.updateViruses()
        self.updateBlobs()
        self.updatePlayers()
        self.updateHashTables()
        self.mergePlayerCells()
        self.checkOverlaps()
        self.spawnStuff()
        

    def updateViruses(self):
        for virus in self.viruses:
            virus.updateMomentum()
            virus.updatePos(self.size, self.size)

    def updateBlobs(self):
        notMovingBlobs = []
        for blob in self.blobs:
            if blob.getSplitVelocityCounter() == 0:
                notMovingBlobs.append(blob)
                continue
            blob.updateMomentum()
            blob.updatePos(self.size, self.size)
        for blob in notMovingBlobs:
            self.blobs.remove(blob)
            self.blobHashTable.deleteObject(blob)
            self.addPellet(blob)

    def updatePlayers(self):
        for player in self.players:
            if player.getIsAlive():
                player.update(self.size, self.size)
                self.performEjections(player)
                self.handlePlayerCollisions(player)
            else:
                player.updateRespawnTime()

    def updateHashTables(self):
        self.playerHashTable.clearBuckets()
        for player in self.players:
            if player.getIsAlive():
                playerCells = player.getCells()
                self.playerHashTable.insertAllObjects(playerCells)

        self.blobHashTable.clearBuckets()
        self.blobHashTable.insertAllObjects(self.blobs)

        self.virusHashTable.clearBuckets()
        self.virusHashTable.insertAllObjects(self.viruses)

    def performEjections(self, player):
        for cell in player.getCells():
            if cell.getBlobToBeEjected():
                blobSpawnPos = cell.eject()
                # Blobs are given a player such that cells of player who eject them don't instantly reabsorb them
                blob = Cell(blobSpawnPos[0], blobSpawnPos[1], EJECTEDBLOB_BASE_MASS * 0.8, None)

                blob.setColor(player.getColor())
                #blob.setEjecterPlayer(player)
                blob.addMomentum(player.getCommandPoint(), self.size, self.size, cell)

                self.addBlob(blob)
                blob.setEjecterCell(cell)


    def handlePlayerCollisions(self, player):
        for cell in player.getCells():
            if cell.justEjected():
                continue
            for otherCell in player.getCells():
                if cell is otherCell or otherCell.justEjected() or (cell.canMerge() and otherCell.canMerge()):
                    continue
                distance = numpy.sqrt(cell.squaredDistance(otherCell))
                summedRadii = cell.getRadius() + otherCell.getRadius()
                if distance < summedRadii and distance != 0:
                    self.adjustCellPositions(cell, otherCell, distance, summedRadii)

    def adjustCellPositions(self, cell1, cell2, distance, summedRadii):
        if cell1.getMass() > cell2.getMass():
            biggerCell = cell1
            smallerCell = cell2
        else:
            biggerCell = cell2
            smallerCell = cell1
        bigPos = biggerCell.getPos()
        smallPos = smallerCell.getPos()
        distanceScaling = (summedRadii - distance) / distance
        massDifferenceScaling = smallerCell.getMass() /  biggerCell.getMass()
        xDiffScaled = (bigPos[0] - smallPos[0]) * distanceScaling
        yDiffScaled = (bigPos[1] - smallPos[1]) * distanceScaling
        newXBigCell = bigPos[0] + xDiffScaled * massDifferenceScaling
        newYBigCell = bigPos[1] + yDiffScaled * massDifferenceScaling
        newXSmallCell = smallPos[0] - xDiffScaled * (1 - massDifferenceScaling)
        newYSmallCell = smallPos[1] - yDiffScaled * (1 - massDifferenceScaling)
        newPosBig = [newXBigCell, newYBigCell]
        newPosSmall = [newXSmallCell, newYSmallCell]
        self.adjustCellPos(biggerCell, newPosBig)
        self.adjustCellPos(smallerCell, newPosSmall)

    def mergePlayerCells(self):
        for player in self.players:
            if player.getIsAlive():
                cells = player.getMergableCells()
                if len(cells) > 1:
                    cells.sort(key = lambda p: p.getMass(), reverse = True)
                    for cell1 in cells:
                        if not cell1.isAlive():
                            continue
                        for cell2 in cells:
                            if (not cell2.isAlive()) or (cell2 is cell1):
                                continue
                            if cell1.overlap(cell2):
                                self.mergeCells(cell1, cell2)
                                if not cell1.isAlive():
                                    break

    def checkOverlaps(self):
        self.virusBlobOverlap()
        self.playerVirusOverlap()
        self.playerPelletOverlap()
        self.playerBlobOverlap()
        self.playerPlayerOverlap()

    def playerPelletOverlap(self):
        for player in self.players:
            if player.getIsAlive():
                for cell in player.getCells():
                    for pellet in self.pelletHashTable.getNearbyObjects(cell):
                        if cell.overlap(pellet) and cell.canEat(pellet):
                            self.eatPellet(cell, pellet)

    def playerBlobOverlap(self):
        for player in self.players:
            if player.getIsAlive():
                for cell in player.getCells():
                    for blob in self.blobHashTable.getNearbyObjects(cell):
                        # If the ejecter player's cell is not the one overlapping with blob
                        if cell.overlap(blob) and blob.getEjecterCell() is not cell and cell.canEat(blob):
                            self.eatBlob(cell, blob)


    def playerVirusOverlap(self):
        for player in self.players:
            if player.getIsAlive():
                for cell in player.getCells():
                    for virus in self.virusHashTable.getNearbyObjects(cell):
                        if cell.overlap(virus) and cell.getMass() > 1.25 * virus.getMass():
                            self.eatVirus(cell, virus)

    def playerPlayerOverlap(self):
        for player in self.players:
            if player.getIsAlive():
                for playerCell in player.getCells():
                    opponentCells = self.playerHashTable.getNearbyEnemyObjects(playerCell)                
                    for opponentCell in opponentCells:
                            if playerCell.overlap(opponentCell):
                                if playerCell.canEat(opponentCell):
                                    self.eatPlayerCell(playerCell, opponentCell)
                                elif opponentCell.canEat(playerCell):
                                    self.eatPlayerCell(opponentCell, playerCell)
                                    break

    def virusBlobOverlap(self):
        # After 7 feedings the virus splits in roughly the opposite direction of the last incoming ejectable
        # The ejected viruses bounce off of the edge of the fields
        for virus in self.viruses:
            nearbyBlobs = self.blobHashTable.getNearbyObjects(virus)
            for blob in nearbyBlobs:
                if virus.overlap(blob):
                    self.virusEatBlob(virus, blob)


    def spawnStuff(self):
        self.spawnPellets()
        if self.virusEnabled:
            self.spawnViruses()
        self.spawnPlayers()

    def spawnViruses(self):
        while len(self.viruses) < self.maxVirusCount:
            self.spawnVirus()

    def spawnVirus(self):
        xPos, yPos = self.getSpawnPos(VIRUS_BASE_RADIUS)
        acceptableSpawnRange = HASH_BUCKET_SIZE - VIRUS_BASE_RADIUS
        xPos += numpy.random.randint((-1)*acceptableSpawnRange/2, acceptableSpawnRange/2)
        yPos += numpy.random.randint((-1)*acceptableSpawnRange/2, acceptableSpawnRange/2)
        size = VIRUS_BASE_SIZE
        virus = Cell(xPos, yPos, size, None)
        virus.setName("Virus")
        virus.setColor((0,255,0))
        self.addVirus(virus)

    def spawnPlayers(self):
        for player in self.deadPlayers[:]:
            if player.getRespawnTime() == 0:
                self.deadPlayers.remove(player)
                self.initializePlayer(player)

    def getSpawnPos(self, radius):
        cols = self.playerHashTable.getCols()
        totalBuckets = self.playerHashTable.getRows() * cols
        spawnBucket = numpy.random.randint(0, totalBuckets)
        count = 0
        while self.playerHashTable.getBuckets()[spawnBucket] and count < totalBuckets:
            spawnBucket = (spawnBucket + 1) % totalBuckets
            count += 1
        if count == totalBuckets:
            xPos = numpy.random.randint(0, self.size)
            yPos = numpy.random.randint(0, self.size)
        else:
            x = spawnBucket % cols
            y = (spawnBucket - x) / cols
            left = (x - 1)  * HASH_BUCKET_SIZE
            top =  y * HASH_BUCKET_SIZE
            xPos = numpy.random.randint(left + radius, left + HASH_BUCKET_SIZE - radius)
            yPos = numpy.random.randint(top + radius, top + HASH_BUCKET_SIZE - radius)
        return xPos, yPos

    def spawnPellets(self):
        while len(self.pellets) < self.maxCollectibleCount:
            self.spawnPellet()

    def spawnPellet(self):
        xPos = numpy.random.randint(0, self.size)
        yPos = numpy.random.randint(0, self.size)
        size = randomSize()
        pellet = Cell(xPos, yPos, size, None)
        pellet.setName("Pellet")
        self.addPellet(pellet)

    # Cell1 eats Cell2. Therefore Cell1 grows and Cell2 is deleted
    def virusEatBlob(self, virus, blob):
        self.eatCell(virus, self.virusHashTable, blob, self.blobHashTable, self.blobs)
        if virus.getMass() >= VIRUS_BASE_SIZE + 7 * EJECTEDBLOB_BASE_MASS * 0.8:
            oppositeX = 2 * virus.getPos()[0] - blob.getPos()[0]
            oppositeY = 2 * virus.getPos()[1] - blob.getPos()[1]
            oppositePoint = [oppositeX, oppositeY]
            newVirus = virus.split(oppositePoint, self.size, self.size)
            newVirus.setColor(virus.getColor())
            newVirus.setName(virus.getName())
            self.addVirus(newVirus)

    def eatPellet(self, playerCell, pellet):
        self.eatCell(playerCell, self.playerHashTable, pellet, self.pelletHashTable, self.pellets)

    def eatBlob(self, playerCell, blob):
        self.eatCell(playerCell, self.playerHashTable, blob, self.blobHashTable, self.blobs)

    def eatVirus(self, playerCell, virus):
        self.eatCell(playerCell, self.playerHashTable, virus, self.virusHashTable, self.viruses, True)
        self.playerCellAteVirus(playerCell)

    def eatCell(self, eatingCell, eatingCellHashtable, cell, cellHashtable, cellList, isVirus = None):
        mass = cell.getMass()
        if isVirus:
            mass *= VIRUS_EAT_FACTOR
        adjustCellSize(eatingCell, mass, eatingCellHashtable)
        cellList.remove(cell)
        cellHashtable.deleteObject(cell)
        cell.setAlive(False)

    def eatPlayerCell(self, largerCell, smallerCell):
        adjustCellSize(largerCell, smallerCell.getMass(), self.playerHashTable)
        self.deletePlayerCell(smallerCell)

    def playerCellAteVirus(self, playerCell):
        player = playerCell.getPlayer()
        numberOfCells = len(player.getCells())
        numberOfNewCells = 16 - numberOfCells
        if numberOfNewCells == 0:
            return
        distributedMass = playerCell.getMass() * VIRUS_EXPLOSION_CELL_MASS_PROPORTION
        massPerCell = distributedMass / numberOfNewCells
        playerCell.resetMergeTime(MERGE_TIME_VIRUS_FACTOR)
        adjustCellSize(playerCell, -1 * massPerCell * numberOfNewCells, self.playerHashTable)
        for cellIdx in range(numberOfNewCells):
            cellPos = playerCell.getPos()
            newCell = Cell(cellPos[0], cellPos[1], massPerCell, player)
            cellAngle = numpy.deg2rad(numpy.random.randint(0,360))
            xPoint = math.cos(cellAngle) * playerCell.getRadius() * 12 + cellPos[0]
            yPoint = math.sin(cellAngle) * playerCell.getRadius() * 12 + cellPos[1]
            movePoint = (xPoint, yPoint)
            newCell.setMoveDirection(movePoint)
            newCell.addMomentum(movePoint, self.size, self.size, playerCell)
            newCell.resetMergeTime(0.8)
            self.addPlayerCell(newCell)

    def mergeCells(self, firstCell, secondCell):
        if firstCell.getMass() > secondCell.getMass():
            biggerCell = firstCell
            smallerCell = secondCell
        else:
            biggerCell = secondCell
            smallerCell = firstCell
        adjustCellSize(biggerCell, smallerCell.getMass(), self.playerHashTable)
        self.deletePlayerCell(smallerCell)

    def deletePlayerCell(self, playerCell):
        self.playerHashTable.deleteObject(playerCell)
        player = playerCell.getPlayer()
        player.removeCell(playerCell)
        if not player.getCells():
            self.deadPlayers.append(player)
            player.setDead()

    def addPellet(self, pellet):
        self.pelletHashTable.insertObject(pellet)
        self.pellets.append(pellet)

    def addBlob(self, blob):
        #self.blobHashTable.insertObject(blob)
        self.blobs.append(blob)

    def addVirus(self, virus):
        #self.virusHashTable.insertObject(virus)
        self.viruses.append(virus)

    def addPlayerCell(self, playerCell):
        self.playerHashTable.insertObject(playerCell)
        playerCell.getPlayer().addCell(playerCell)

    def adjustCellPos(self, cell, newPos):
        #hashtable.deleteObject(cell)
        x = min(self.size, max(0, newPos[0]))
        y = min(self.size, max(0, newPos[1]))
        cell.setPos([x,y])
        #hashtable.insertObject(cell)

    # Setters:
    def addPlayer(self, player):
        player.setAlive()
        self.players.append(player)

    # Getters:
    def getVirusEnabled(self):
        return self.virusEnabled

    @staticmethod
    def getPortionOfCellsInFov(cells, fovPos, fovSize):
        return [cell for cell in cells if cell.isInFov(fovPos,fovSize)]

    def getPlayerCellsInFov(self, fovPos, fovSize):
        cellsNearFov = self.getCellsFromHashTableInFov(self.playerHashTable, fovPos, fovSize)
        return self.getPortionOfCellsInFov(cellsNearFov, fovPos, fovSize)

    def getFoVPlayerCellsInFov(self, fovPlayer):
        playerCellsInFov = self.getPlayerCellsInFov(fovPlayer.getFovPos(), fovPlayer.getFovSize())
        return [cell for cell in playerCellsInFov if cell.getPlayer() is fovPlayer]

    def getEnemyPlayerCellsInFov(self, fovPlayer):
        playerCellsInFov = self.getPlayerCellsInFov(fovPlayer.getFovPos(), fovPlayer.getFovSize())
        return [cell for cell in playerCellsInFov if cell.getPlayer() is not fovPlayer]

    def getEnemyPlayerCellsInGivenFov(self, fovPlayer, fovPos, fovSize):
        playerCellsInFov = self.getPlayerCellsInFov(fovPos, fovSize)
        return [cell for cell in playerCellsInFov if cell.getPlayer() is not fovPlayer]

    def getPelletsInFov(self, fovPos, fovSize):
        pelletsNearFov = self.getCellsFromHashTableInFov(self.pelletHashTable, fovPos, fovSize)
        return self.getPortionOfCellsInFov(pelletsNearFov, fovPos, fovSize)

    def getVirusesInFov(self, fovPos, fovSize):
        virusesNearFov = self.getCellsFromHashTableInFov(self.virusHashTable, fovPos, fovSize)
        return self.getPortionOfCellsInFov(virusesNearFov, fovPos, fovSize)
    
    def getBlobsInFov(self, fovPos, fovSize):
        blobsNearFov = self.getCellsFromHashTableInFov(self.blobHashTable, fovPos, fovSize)
        return self.getPortionOfCellsInFov(blobsNearFov, fovPos, fovSize)

    @staticmethod
    def getCellsFromHashTableInFov(hashtable, fovPos, fovSize):
        return hashtable.getNearbyObjectsInArea(fovPos, fovSize / 2)
    
    def getWidth(self):
        return self.size

    def getHeight(self):
        return self.size

    def getPellets(self):
        return self.pellets

    def getBlobs(self):
        return self.blobs

    def getViruses(self):
        return self.viruses

    def getPlayerCells(self):
        cells = []
        for player in self.players:
            cells += player.getCells()
        return cells

    def getDeadPlayers(self):
        return self.deadPlayers

    def getPlayers(self):
        return self.players

    @staticmethod
    def getReward(player):
        return player.getTotalMass()
