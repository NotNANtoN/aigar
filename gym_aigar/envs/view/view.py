import pygame
import numpy
import math
import os
from pygame import gfxdraw

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


def softmax(values):
    maxVal = max(values)
    shiftedVals = [value - maxVal for value in values]
    distribution_values = [math.e ** value for value in shiftedVals]
    distSum = 1#distSum = numpy.sum(distribution_values)
    return [value / distSum for value in distribution_values]


class View:
    def __init__(self, model, width, height, parameters):
        self.width = width
        self.height = height
        self.parameters = parameters
        self.screenDims = None
        self.model = model
        self.numberOfScreens = None
        self.model.register_listener(self.model_event)
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0,30)
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.splitScreen = False
        self.playerScreens = []
        self.setNumberOfScreens()
        pygame.init()
        pygame.display.set_caption('A.I.gar')

        # Rendering fonts for the leaderboard and initializing it
        numbOfPlayers = min(10, len(model.getPlayers()))
        self.leaderBoardTextHeight = 25
        self.leaderBoardTitleHeight = 28
        self.leaderBoardFont = pygame.font.SysFont(None, self.leaderBoardTextHeight)
        leaderBoardTitleFont = pygame.font.SysFont(None, self.leaderBoardTitleHeight)
        self.leaderBoardTitle = leaderBoardTitleFont.render("Leaderboard", True, (255, 255, 255))
        self.leaderBoardWidth = self.leaderBoardTitle.get_width() + 15
        self.leaderBoardHeight = self.leaderBoardTitleHeight + self.leaderBoardTextHeight * numbOfPlayers + 2
        self.leaderBoard = pygame.Surface((self.leaderBoardWidth, self.leaderBoardHeight))  # the size of your rect
        self.leaderBoard.set_alpha(128)

    def setNumberOfScreens(self):
        humansNr = len(self.model.getHumans())
        if humansNr > 1:
            self.numberOfScreens = humansNr
            self.screenDims = numpy.array([int((self.width - (humansNr - 1)) / humansNr), self.height])
            self.splitScreen = True
            for playerScreen in range(humansNr):
                screen = pygame.Surface(self.screenDims)
                self.playerScreens.append(screen)
        else:
            self.numberOfScreens = 1
            self.screenDims = numpy.array([self.width, self.height])
            self.playerScreens.append(self.screen)

    def drawDebugInfo(self):
        for screenNr in range(self.numberOfScreens):
            screen = self.playerScreens[screenNr]
            cells = self.model.getPlayerCells()
            fovPos = self.model.getFovPos(screenNr)
            fovSize = self.model.getFovSize(screenNr)
            for player in self.model.getPlayers():
                if player.getIsAlive() and player.getSelected():
                    playerFovPos = player.getFovPos()
                    playerFovSize = player.getFovSize()
                    scaledPos= self.modelToViewScaling(playerFovPos, fovPos, fovSize)
                    scaledSize = self.modelToViewScaleRadius(playerFovSize, fovSize)
                    left = scaledPos[0] - scaledSize / 2
                    right = left + scaledSize
                    top = scaledPos[1] - scaledSize / 2
                    bottom = top + scaledSize
                    topleft = (left, top)
                    topright = (right, top)
                    bottomleft = (left, bottom)
                    bottomright = (right, bottom)
                    pygame.draw.line(screen, BLACK, topleft, topright)
                    pygame.draw.line(screen, BLACK, topright, bottomright)
                    pygame.draw.line(screen, BLACK, bottomright, bottomleft)
                    pygame.draw.line(screen, BLACK, bottomleft, topleft)



            # Print grid for grid state representation
            for bot in self.model.getBots():
                player = bot.getPlayer()

                # Plot command point:
                if player.getIsAlive():
                    commandPoint = player.getCommandPoint()
                    pos = commandPoint[0], commandPoint[1]
                    scaledPos = self.modelToViewScaling(pos, fovPos, fovSize)
                    pygame.gfxdraw.filled_circle(screen, int(scaledPos[0]), int(scaledPos[1]), 5, player.color)

                # Draw Grid lines:
                if player.getIsAlive() and player.getSelected() and bot.getType() == "NN":
                    playerFovPos = player.getFovPos()
                    playerFovSize = player.getFovSize()
                    neededLines = bot.getGridSquaresPerFov() - 1
                    distanceBetweenLines = self.modelToViewScaleRadius(playerFovSize /
                                                                       bot.parameters.GRID_SQUARES_PER_FOV, fovSize)
                    scaledPos = self.modelToViewScaling(playerFovPos, fovPos, fovSize)
                    scaledSize = self.modelToViewScaleRadius(playerFovSize, fovSize)
                    left = scaledPos[0] - scaledSize / 2
                    right = left + scaledSize
                    top = scaledPos[1] - scaledSize / 2
                    bottom = top + scaledSize

                    for xIdx in range(1, neededLines + 1):
                        posUp = (left + xIdx * distanceBetweenLines, top)
                        posDown = (left + xIdx * distanceBetweenLines, bottom )
                        pygame.draw.line(screen, BLACK, posUp, posDown)

                    for yIdx in range(1, neededLines + 1):
                        posLeft = (left, top + yIdx * distanceBetweenLines)
                        posRight = (right, top + yIdx * distanceBetweenLines )
                        pygame.draw.line(screen, BLACK, posLeft, posRight)

                    # draw Q-values:
                    if bot.learningAlg is not None and str(bot.learningAlg) == "Q-learning" \
                            and bot.learningAlg.current_q_values is not None:

                        raw_q_vals = bot.learningAlg.current_q_values
                        q_vals = softmax(raw_q_vals)
                        gridsPerSide = int(math.sqrt(bot.parameters.NUM_ACTIONS))
                        gridSize = scaledSize / gridsPerSide
                        for idx, q_value in enumerate(q_vals):
                            i = idx % gridsPerSide
                            j = idx // gridsPerSide
                            x = left + i * gridSize
                            y = top + j * gridSize
                            greenPart = int(q_value * 255)
                            font = pygame.font.SysFont(None, 65)
                            text = str(round(raw_q_vals[idx], 1))
                            textSurface = font.render(text, True, (0,0,0))
                            textWidth = textSurface.get_width()
                            textHeight = textSurface.get_height()

                            s = pygame.Surface((gridSize, gridSize))  # the size of your rect
                            s.set_alpha(75 + int(q_value * 175))  # alpha level
                            s.fill((255 - greenPart, 255, 255 - greenPart))  # this fills the entire surface
                            s.blit(textSurface, ((gridSize - textWidth) / 2 , (gridSize - textHeight) / 2))
                            screen.blit(s, (x, y))  # (0,0) are the top-left coordinates




            for cell in cells:
                pos = numpy.array(cell.getPos())
                scaledPos = self.modelToViewScaling(pos, fovPos, fovSize)
                pygame.draw.line(self.playerScreens[screenNr], RED, scaledPos.astype(int),
                                 numpy.array(cell.getVelocity()) * 10 +
                                 numpy.array(scaledPos.astype(int)))
                

    def drawCells(self, cells, fovPos, fovSize, screen):
        for cell in cells:
            self.drawSingleCell(cell, fovPos, fovSize, screen)

    def drawSingleCell(self, cell, fovPos, fovSize, screen):
        unscaledRad = cell.getRadius()
        unscaledPos = numpy.array(cell.getPos())
        color = cell.getColor()
        if __debug__ and cell.getPlayer():
            if cell.getPlayer().isExploring():
                color = (0, 255, 0)

        player = cell.getPlayer()
        rad = int(self.modelToViewScaleRadius(unscaledRad, fovSize))
        pos = self.modelToViewScaling(unscaledPos, fovPos, fovSize).astype(int)
        if rad >= 4:
            pygame.gfxdraw.filled_circle(screen, pos[0], pos[1], rad, color)
            if cell.getName() == "Virus":
                # Give Viruses a black surrounding circle
                pygame.gfxdraw.aacircle(screen, pos[0], pos[1], rad, (0,0,0))
            else:
                pygame.gfxdraw.aacircle(screen, pos[0], pos[1], rad, color)
        else:
            # Necessary to avoid that collectibles are drawn as little X's when the fov is huge
            pygame.draw.circle(screen, color, pos, rad)
        if player is not None or (__debug__ and cell.getName() == "Virus"):
            font = pygame.font.SysFont(None, int(rad / 2))
            name = font.render(cell.getName(), True, (0,0,0))
            textPos = [pos[0] - name.get_width() / 2, pos[1] - name.get_height() / 2]
            screen.blit(name, textPos)
            if __debug__:
                mass = font.render("Mass:" + str(int(cell.getMass())), True, (0, 0, 0))
                textPos = [pos[0] - mass.get_width() / 2, pos[1] - mass.get_height() / 2 + name.get_height()]
                screen.blit(mass, textPos)
                if cell.getMergeTime() > 0:
                    text = font.render(str(int(cell.getMergeTime())), True, (0, 0, 0))
                    textPos = [pos[0] - text.get_width() / 2, pos[1] - text.get_height() / 2 + name.get_height() + mass.get_height()]
                    screen.blit(text, textPos)


    def drawAllCells(self):
        for humanNr in range(self.numberOfScreens):
            fovPos = self.model.getFovPos(humanNr)
            fovSize = self.model.getFovSize(humanNr)
            if fovPos is None:
                continue
            pellets = self.model.getField().getPelletsInFov(fovPos, fovSize)
            blobs = self.model.getField().getBlobsInFov(fovPos, fovSize)
            viruses = self.model.getField().getVirusesInFov(fovPos, fovSize)
            playerCells = self.model.getField().getPlayerCellsInFov(fovPos, fovSize)

            allCells = pellets + blobs + viruses + playerCells
            allCells.sort(key = lambda p: p.getMass())

            self.drawCells(allCells, fovPos, fovSize, self.playerScreens[humanNr])


    def drawHumanStats(self):
        if self.model.hasHuman():
            for humanNr in range(self.numberOfScreens):
                totalMass = self.model.getHumans()[humanNr].getTotalMass()
                name = "Total Mass: " + str(int(totalMass))
                font = pygame.font.SysFont(None, int(min(150, 30 + numpy.sqrt(totalMass))))
                text = font.render(name, False, (min(255,int(totalMass / 5)), min(100,int(totalMass / 10)), min(100,int(totalMass / 10))))
                pos = (self.screenDims[0], self.height - text.get_height())
                self.playerScreens[humanNr].blit(text, pos)

    def drawScreenSeparators(self):
        for screenNumber in range(self.numberOfScreens):
            x = self.screenDims[0] * screenNumber + 1
            pygame.gfxdraw.line(self.screen, x, 0, x, self.screenDims[1], BLACK)


    def drawLeaderBoard(self):
        self.leaderBoard.fill((0, 0, 0))
        players = self.model.getTopTenPlayers()
        numberOfPositionsShown = len(players)
        self.leaderBoard.blit(self.leaderBoardTitle, (8, self.leaderBoardTitleHeight / 4))
        for i in range(numberOfPositionsShown):
            currentPlayer = players[i]
            string = str(i + 1) + ". " + currentPlayer.getName() + ": " + str(int(currentPlayer.getTotalMass()))
            text = self.leaderBoardFont.render(string, True, (255, 255, 255))
            pos = (8, self.leaderBoardTitleHeight + i * self.leaderBoardTextHeight)
            self.leaderBoard.blit(text, pos)
        for screen in self.playerScreens:
            screen.blit(self.leaderBoard, (self.screenDims[0] - self.leaderBoardWidth - 10, 10))

    def draw(self):
        self.screen.fill(WHITE)
        if self.splitScreen:
            for screenNr in range(len(self.playerScreens)):
                self.screen.blit(self.playerScreens[screenNr], (self.screenDims[0] * screenNr + screenNr, 0))
                self.playerScreens[screenNr].fill(WHITE)
            self.drawScreenSeparators()
        self.drawAllCells()
        self.drawHumanStats()
        self.drawLeaderBoard()
        if __debug__:
            self.drawDebugInfo()
        pygame.display.update()


    def model_event(self):
        self.draw()

    def modelToViewScaling(self, pos, fovPos, fovSize):
        adjustedPos = pos - fovPos + (fovSize / 2)
        scaledPos = adjustedPos * (self.screenDims / fovSize)
        return scaledPos

    def viewToModelScaling(self, pos, fovPos, fovSize):
        scaledPos = pos / (self.screenDims / fovSize)
        adjustedPos = scaledPos + fovPos - (fovSize / 2)
        return adjustedPos

    def modelToViewScaleRadius(self, rad, fovSize):
        return rad * (self.screenDims[0] / fovSize)

    # Checks:
    def getScreenDims(self):
        return self.screenDims

    def getWindowWidth(self):
        return self.width

    def getWindowHeight(self):
        return self.height

    def getFullRGB(self):
        return pygame.surfarray.array3d(self.screen)

    def closeView(self):
        pygame.display.quit()
        pygame.quit()




