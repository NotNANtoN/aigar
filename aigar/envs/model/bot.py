import numpy

from .parameters import *
from .spatialHashTable import SpatialHashTable


def isCellData(cell):
    return [cell.getX(), cell.getY(), cell.getRadius()]


def checkNan(value):
    if math.isnan(value):
        print("ERROR: predicted reward is nan")
        quit()


def getRelativeCellPos(cell, left, top, size):
    if cell is not None:
        return [round((cell.getX() - left) / size, 5), round((cell.getY() - top) / size, 5)]
    else:
        return [0, 0]


class Bot(object):
    _greedyId = 0
    _nnId = 0
    _randomId = 0
    num_NNbots = 0
    num_Greedybots = 0

    @property
    def randomId(self):
        return type(self)._randomId

    @randomId.setter
    def randomId(self, val):
        type(self)._randomId = val

    @property
    def greedyId(self):
        return type(self)._greedyId

    @greedyId.setter
    def greedyId(self, val):
        type(self)._greedyId = val

    @property
    def nnId(self):
        return type(self)._nnId

    @nnId.setter
    def nnId(self, val):
        type(self)._nnId = val

    def __repr__(self):
        return self.type #+ str(self.id)

    def __init__(self, player, field, bot_type, learningAlg, parameters, rgbGenerator=None, use_enemy_grid=True):
        if bot_type == "Greedy":
            self.id = self.greedyId
            self.greedyId += 1
        elif bot_type == "NN":
            self.id = self.nnId
            self.nnId += 1
        elif bot_type == "Random":
            self.id = self.randomId
            self.randomId += 1
            
        self.use_enemy_grid = use_enemy_grid
        self.num_grids = 3
        if self.use_enemy_grid:
            self.num_grids += 1
        if field.getVirusEnabled():
            self.virus_enabled = True
            self.num_grids += 1
        else:
            self.virus_enabled = False


        # TODO: What is the difference between memories and experiences?
        self.experiences = []
        self.rgbGenerator = rgbGenerator
        #self.gatherExperiences = parameters.GATHER_EXP
        self.parameters = parameters
        self.lastMass = None
        self.lastReward = None
        self.lastAction = None
        self.fovSize = None
        self.lastFovSize = None
        self.currentAction = None
        self.gridSquaresPerFov = GRID_SQUARES_PER_FOV

        self.type = bot_type
        self.player = player
        self.field = field
        self.time = 0
        if self.type == "Greedy":
            self.splitLikelihood = numpy.random.randint(9950, 10000)
            self.ejectLikelihood = 100000  # numpy.random.randint(9990,10000)
        self.totalMasses = []
        self.memories = []
        self.secondLastSelfGrid = None
        self.lastSelfGrid = None
        self.secondLastEnemyGrid = None
        self.lastEnemyGrid = None
        self.lastAllPlayerGrid = None
        self.lastPixelGrid = None
        
        self.reset()


    def resetMassList(self):
        self.totalMasses = []

    def reset(self):
        self.lastMass = None
        self.oldState = None
        self.lastMemory = None
        self.skipFrames = 0
        self.cumulativeReward = 0
        self.lastReward = 0
        self.rewardAvgOfEpisode = 0
        self.rewardLenOfEpisode = 0
        self.currentlySkipping = False
        
        if self.type == "Greedy" or self.type == "Random":
            self.currentAction = [0, 0, 0, 0]

        self.experiences = []

    def updateRewards(self):
        self.cumulativeReward += self.getReward() if self.lastMass else 0
        self.lastReward = self.cumulativeReward

    # Returns true if we skip this frame
    def updateFrameSkip(self):
        # Do not train if we are skipping this frame
        if self.skipFrames > 0:
            self.skipFrames -= 1
            self.latestTDerror = None
            if self.player.getIsAlive():
                return True
        return False

    def updateValues(self, extraInfo, newAction, newState, newLastMemory=None):
        if newLastMemory is not None:
            self.lastMemory = newLastMemory
        # Reset frame skipping variables
        self.cumulativeReward = 0
        self.skipFrames = self.parameters.FRAME_SKIP_RATE
        self.oldState = newState
        self.lastAction = self.currentAction
        self.currentAction = newAction


    def setExploring(self, val):
        self.player.setExploring(val)


    def setMassesOverTime(self, array):
        self.totalMasses = array


    def make_random_bot_move(self):
        if self.time % self.parameters.FRAME_SKIP_RATE == 0:
            self.currentAction[0] = numpy.random.random()
            self.currentAction[1] = numpy.random.random()
            self.currentAction[2] = numpy.random.random() if ENABLE_SPLIT else False
            self.currentAction[3] = numpy.random.random() if ENABLE_EJECT else False
        self.time += 1

    def makeMove(self, action=None):
        self.totalMasses.append(self.player.getTotalMass())

        if not self.player.getIsAlive():
            return

        if self.type == "Greedy":
            self.make_greedy_bot_move()

        if self.type == "Random":
            self.make_random_bot_move()
        
        if self.type == "Gym":
            self.currentAction = numpy.zeros(4)
            self.currentAction[:len(action)] = action
            
        if self.player.getIsAlive():
            self.lastMass = self.player.getTotalMass()

        action_taken = list(self.currentAction)
        if self.currentlySkipping:
            action_taken[2:] = [0, 0]
        self.set_command_point(action_taken)


    def getStateRepresentation(self):
        stateRepr = None
        if self.player.getIsAlive():
            if self.parameters.GRID_VIEW_ENABLED:
                if self.parameters.CNN_REPR:
                    if self.parameters.CNN_P_REPR:
                        rgb_values = self.rgbGenerator.get_cnn_inputRGB(self.player)
                        stateRepr = (rgb_values - 255) / 100  # Normalize input to range [0,1]
                        if self.parameters.CNN_LAST_GRID:
                            stateRepr = numpy.concatenate((stateRepr, self.lastPixelGrid), axis=2)
                            self.lastPixelGrid = stateRepr
                    else:
                        stateRepr = self.getGridStateRepresentation()
                else:
                    gridView = self.getGridStateRepresentation()
                    gridView = gridView.flatten()
                    if self.parameters.EXTRA_INPUT:
                        additionalFeatures = self.getAdditionalFeatures()
                        stateRepr = numpy.concatenate((gridView, additionalFeatures))
                    else:
                        stateRepr = gridView
                    shape = [1]
                    shape.extend(numpy.shape(stateRepr))
                    stateRepr = stateRepr.reshape(shape)
            else:
                stateRepr = self.getSimpleStateRepresentation()

        return stateRepr


    def getAdditionalFeatures(self):
        additionalFeatures = []
        if self.parameters.USE_LAST_FOVSIZE:
            self.lastFovSize = self.fovSize
            additionalFeatures.append(self.lastFovSize)
        if self.parameters.USE_FOVSIZE:
            self.fovSize = self.player.getFovSize()
            additionalFeatures.append(self.fovSize)
        if self.parameters.USE_TOTALMASS:
            mass = self.player.getTotalMass()
            additionalFeatures.append(mass)
        if self.parameters.USE_LAST_ACTION:
            last_action = self.currentAction if self.currentAction is not None else [0, 0, 0, 0]
            additionalFeatures.extend(last_action)
            if len(last_action) < 4:
                additionalFeatures.extend(numpy.zeros(4 - len(last_action)))
        if self.parameters.USE_SECOND_LAST_ACTION:
            second_last_action = self.lastAction if self.lastAction is not None else [0, 0, 0, 0]
            additionalFeatures.extend(second_last_action)
            if len(second_last_action) < 4:
                additionalFeatures.extend(numpy.zeros(4 - len(second_last_action)))
        return additionalFeatures


    def getObsSize(self):
        return (self.gridSquaresPerFov, self.gridSquaresPerFov, self.num_grids)

    def getGridStateRepresentation(self):
        # Get Fov infomation
        fieldSize = self.field.getWidth()
        fovSize = self.player.getFovSize()
        fovPos = self.player.getFovPos()
        x = fovPos[0]
        y = fovPos[1]
        left = x - fovSize / 2
        top = y - fovSize / 2
        # Initialize spatial hash tables:
        gridSquaresPerFov = self.gridSquaresPerFov
        gsSize = fovSize / gridSquaresPerFov  # (gs = grid square)

        pelletSHT = SpatialHashTable(fovSize, gsSize, left, top)  # SHT = spatial hash table
        totalPellets = self.field.getPelletsInFov(fovPos, fovSize)
        pelletSHT.insertAllFloatingPointObjects(totalPellets)

        enemyCells = self.field.getEnemyPlayerCellsInFov(self.player)
        playerCells = self.field.getPortionOfCellsInFov(self.player.getCells(), fovPos, fovSize)
        allPlayerSHT = None
        playerSHT = None
        enemySHT = None
        
        playerSHT = SpatialHashTable(fovSize, gsSize, left, top)
        playerSHT.insertAllFloatingPointObjects(playerCells)
        enemySHT = SpatialHashTable(fovSize, gsSize, left, top)
        enemySHT.insertAllFloatingPointObjects(enemyCells)

        virusSHT = None
        if self.field.getVirusEnabled():
            virusSHT = SpatialHashTable(fovSize, gsSize, left, top)
            virusCells = self.field.getVirusesInFov(fovPos, fovSize)
            virusSHT.insertAllFloatingPointObjects(virusCells)

        # Calculate mass of biggest cell:
        #if self.parameters.NORMALIZE_GRID_BY_MAX_MASS:
        #    allCells = numpy.concatenate((playerCells, enemyCells))
        #    biggestCellMass = max(allCells, key = lambda cell: cell.getMass()).getMass()

        # Initialize grid squares with zeros:
        gsPelletMass = numpy.zeros((gridSquaresPerFov, gridSquaresPerFov))
        gsWalls = numpy.zeros((gridSquaresPerFov, gridSquaresPerFov))
        gsBiggestAllCellMass = None
        gsBiggestEnemyCellMass = None
        gsBiggestOwnCellMass = None
        #if self.parameters.ALL_PLAYER_GRID:
        #    gsBiggestAllCellMass = numpy.zeros((gridSquaresPerFov, gridSquaresPerFov))
        #else:
        gsBiggestEnemyCellMass = numpy.zeros((gridSquaresPerFov, gridSquaresPerFov))
        gsBiggestOwnCellMass = numpy.zeros((gridSquaresPerFov, gridSquaresPerFov))
        gsVirus = None
        if self.virus_enabled:
            gsVirus = numpy.zeros((gridSquaresPerFov, gridSquaresPerFov))

        gridView = numpy.zeros(self.getObsSize())
        # gsMidPoint is adjusted in the loops
        gsMidPoint = [left + gsSize / 2, top + gsSize / 2]
        for c in range(gridSquaresPerFov):
            for r in range(gridSquaresPerFov):
                count = r + c * gridSquaresPerFov
                # Only check for cells if the grid square fov is within the playing field
                if not (gsMidPoint[0] + gsSize / 2 < 0 or gsMidPoint[0] - gsSize / 2 > fieldSize or
                        gsMidPoint[1] + gsSize / 2 < 0 or gsMidPoint[1] - gsSize / 2 > fieldSize):
                    # Create pellet representation
                    # Make the visionGrid's pellet count a percentage so that the network doesn't have to
                    # work on interpreting the number of pellets relative to the size (and Fov) of the player
                    pelletMassSum = 0
                    pelletsInGS = pelletSHT.getBucketContent(count)
                    if pelletsInGS:
                        for pellet in pelletsInGS:
                            pelletMassSum += pellet.getMass()
                            if NORMALIZE_GRID_BY_MAX_MASS:
                                pelletMassSum /= biggestCellMass
                        gsPelletMass[c][r] = pelletMassSum
                    # Create Enemy Cell mass representation
                    # Make the visionGrid's enemy cell representation a percentage. The player's cell mass
                    # in proportion to the biggest enemy cell's mass in each grid square.
                    enemiesInGS = enemySHT.getBucketContent(count)
                    if enemiesInGS:
                        biggestEnemyInGSMass = max(enemiesInGS, key=lambda cell: cell.getMass()).getMass()
                        if NORMALIZE_GRID_BY_MAX_MASS:
                            biggestEnemyInGSMass /= biggestCellMass
                        gsBiggestEnemyCellMass[c][r] = biggestEnemyInGSMass

                    # Create Own Cell mass representation
                    playerCellsInGS = playerSHT.getBucketContent(count)
                    if playerCellsInGS:
                        biggestFriendInGSMass = max(playerCellsInGS, key=lambda cell: cell.getMass()).getMass()
                        if NORMALIZE_GRID_BY_MAX_MASS:
                            biggestFriendInGSMass /= biggestCellMass
                        gsBiggestOwnCellMass[c][r] = biggestFriendInGSMass

                    # Create Virus Cell representation
                    if self.field.getVirusEnabled():
                        virusesInGS = virusSHT.getBucketContent(count)
                        if virusesInGS:
                            biggestVirus = max(virusesInGS, key=lambda virus: virus.getRadius()).getMass()
                            if NORMALIZE_GRID_BY_MAX_MASS:
                                biggestVirus /= biggestCellMass
                            gsVirus[c][r] = biggestVirus

                # Create Wall representation
                # Calculate how much of the grid square is covered by walls
                leftBorder = min(max(gsMidPoint[0] - gsSize / 2, 0), fieldSize)
                topBorder = min(max(gsMidPoint[1] - gsSize / 2, 0), fieldSize)
                rightBorder = max(min(gsMidPoint[0] + gsSize / 2, fieldSize), 0)
                bottomBorder = max(min(gsMidPoint[1] + gsSize / 2, fieldSize), 0)
                freeArea = (rightBorder - leftBorder) * (bottomBorder - topBorder)
                gsWalls[c][r] = round(1 - (freeArea / (gsSize ** 2)), 3)

                # Increment grid square position horizontally
                gsMidPoint[0] += gsSize
            # Reset horizontal grid square, increment grid square position
            gsMidPoint[0] = left + gsSize / 2
            gsMidPoint[1] += gsSize

        count = 0
        if PELLET_GRID:
            gridView[:, :, count] = gsPelletMass
            count += 1
        if SELF_GRID:
            gridView[:, :, count] = gsBiggestOwnCellMass
            count += 1
        if WALL_GRID:
            gridView[:, :, count] = gsWalls
            count += 1
        if self.use_enemy_grid:
            gridView[:, :, count] = gsBiggestEnemyCellMass
            count += 1
        if self.virus_enabled:
            gridView[:, :, count] = gsVirus
            count += 1            
        return gridView


    def getCoorConvGrids(self):
        dims = (self.gridSquaresPerFov, self.gridSquaresPerFov)
        coordConv_row = numpy.zeros(dims)
        coordConv_col = numpy.zeros(dims)
        for c in range(dims[0]):
            for r in range(dims[1]):
                coordConv_row[c][r] = c
                coordConv_col[c][r] = r
        return coordConv_row, coordConv_col


    def getSimpleStateRepresentation(self):
        # Get data about the field of view of the player
        size = self.player.getFovSize()
        midPoint = self.player.getFovPos()
        x = int(midPoint[0])
        y = int(midPoint[1])
        left = x - int(size / 2)
        top = y - int(size / 2)
        size = int(size)
        # At the moment we only care about the first cell of the current player, to be extended once we get this working
        firstPlayerCell = self.player.getCells()[0]

        # Adding all the state data to totalInfo
        totalInfo = []
        # Add data about player cells
        cellInfos = self.getCellDataOwnPlayer(left, top, size)
        for info in cellInfos:
            totalInfo += info
        # Add data about the closest enemy cell
        playerCellsInFov = self.field.getEnemyPlayerCellsInFov(self.player)
        closestEnemyCell = min(playerCellsInFov,
                               key=lambda p: p.squaredDistance(firstPlayerCell)) if playerCellsInFov else None
        totalInfo += self.isRelativeCellData(closestEnemyCell, left, top, size)
        # Add data about the closest pellet
        pelletsInFov = self.field.getPelletsInFov(midPoint, size)
        closestPellet = min(pelletsInFov, key=lambda p: p.squaredDistance(firstPlayerCell)) if pelletsInFov else None
        closestPelletPos = getRelativeCellPos(closestPellet, left, top, size)
        totalInfo += closestPelletPos
        # Add data about distances to the visible edges of the field
        width = self.field.getWidth()
        height = self.field.getHeight()
        distLeft = x / size if left <= 0 else 1
        distRight = (width - x) / size if left + size >= width else 1
        distTop = y / size if top <= 0 else 1
        distBottom = (height - y) / size if top + size >= height else 1
        totalInfo += [distLeft, distRight, distTop, distBottom]
        return totalInfo


    def set_command_point(self, action):
        midPoint = self.player.getFovPos()
        size = self.player.getFovSize()
        x = int(midPoint[0])
        y = int(midPoint[1])
        left = x - int(size / 2)
        top = y - int(size / 2)
        size = int(size)
        xChoice = left + action[0] * size
        yChoice = top + action[1] * size
        splitChoice = None
        ejectChoice = None
        if len(action) > 2:
            if len(action) == 3:
                if ENABLE_SPLIT:
                    ejectChoice = False
                    splitChoice = True if action[2] > 0.5 else False
                elif ENABLE_EJECT:
                    ejectChoice = True if [3] > 0.5 else False
                    splitChoice = False
            else:
                splitChoice = True if action[2] > 0.5 else False
                ejectChoice = True if action[3] > 0.5 else False
        else:
            splitChoice = False
            ejectChoice = False

        self.player.setCommands(xChoice, yChoice, splitChoice, ejectChoice)

    def make_greedy_bot_move(self):
        midPoint = self.player.getFovPos()
        size = self.player.getFovSize()
        x = int(midPoint[0])
        y = int(midPoint[1])
        left = x - int(size / 2)
        top = y - int(size / 2)

        cellsInFov = self.field.getPelletsInFov(midPoint, size)
        playerCells = self.player.getCells()
        biggestPlayerCell = max(playerCells, key=lambda p: p.getMass())
        # If the bot is split, use its biggest cell as reference
        if len(playerCells) > 1:
            biggestCellMass = biggestPlayerCell.getMass()
            for playerCell in playerCells:
                playerCellMass = playerCell.getMass()
                if biggestCellMass < playerCellMass:
                    biggestPlayerCell = playerCell
                    biggestCellMass = playerCellMass

        playerCellsInFov = self.field.getEnemyPlayerCellsInFov(self.player)
        for opponentCell in playerCellsInFov:
            # If the single celled bot can eat the opponent cell add it to list
            if biggestPlayerCell.getMass() > 1.25 * opponentCell.getMass():
                cellsInFov.append(opponentCell)

        if self.field.getVirusEnabled():
            virusCellsInFov = self.field.getVirusesInFov(midPoint, size)
            for virus in virusCellsInFov:
                # If the single celled bot can eat the opponent cell add it to list
                if biggestPlayerCell.getMass() > 1.25 * virus.getMass():
                    cellsInFov.append(virus)

        if cellsInFov:
            bestCell = max(cellsInFov, key=lambda p: p.getMass() / (
                p.squaredDistance(biggestPlayerCell) if p.squaredDistance(biggestPlayerCell) != 0 else 1))
            bestCellPos = getRelativeCellPos(bestCell, left, top, size)
            self.currentAction[0] = bestCellPos[0]
            self.currentAction[1] = bestCellPos[1]
        else:
            size = int(size / 2)
            self.currentAction[0] = numpy.random.random()
            self.currentAction[1] = numpy.random.random()
        self.currentAction[2] = False
        self.currentAction[3] = False
        if ENABLE_GREEDY_SPLIT:
            randNumSplit = numpy.random.randint(0, 10000)
            randNumEject = numpy.random.randint(0, 10000)
            if randNumSplit > self.splitLikelihood:
                self.currentAction[2] = True
            if randNumEject > self.ejectLikelihood:
                self.currentAction[3] = True
        else:
            self.currentAction[2] = False
            self.currentAction[3] = False

    def isRelativeCellData(self, cell, left, top, size):
        return getRelativeCellPos(cell, left, top, size) + \
               ([round(cell.getRadius() / size if cell.getRadius() <= size else 1, 5)] if cell is not None else [0])

    def getMassOverTime(self):
        return self.totalMasses

    def getAvgReward(self):
        return self.rewardAvgOfEpisode

    def getLastReward(self):
        return self.lastReward

    def getCellDataOwnPlayer(self, left, top, size):
        cells = self.player.getCells()
        totalCells = len(cells)
        return [self.isRelativeCellData(cells[idx], left, top, size) if idx < totalCells else [0, 0, 0]
                for idx in range(1)]

    def getReward(self):
        if self.lastMass is None:
            return 0
        if not self.player.getIsAlive():
            reward = -1 * self.lastMass
        else:
            currentMass = self.player.getTotalMass()
            reward = currentMass - self.lastMass
        return reward 

    def getFrameSkipRate(self):
        return self.parameters.FRAME_SKIP_RATE

    def getType(self):
        return self.type

    def getPlayer(self):
        return self.player

    def getTrainMode(self):
        return self.trainMode

    def getLastState(self):
        return self.oldState

    def getCurrentActionIdx(self):
        return self.currentActionIdx

    def getCurrentAction(self):
        return self.currentAction

    def getCumulativeReward(self):
        return self.cumulativeReward

    def getLastMemory(self):
        return self.lastMemory

    def getExpRepEnabled(self):
        return self.parameters.EXP_REPLAY_ENABLED

    def getGridSquaresPerFov(self):
        return self.getGridSquaresPerFov()

    def getExperiences(self):
        return self.experiences

