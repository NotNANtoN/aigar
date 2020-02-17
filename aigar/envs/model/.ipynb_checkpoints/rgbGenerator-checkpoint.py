import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
import numpy
from pygame import gfxdraw

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class RGBGenerator:
    def __init__(self, field, parameters):
        self.field = field
        self.parameters = parameters

        self.length = 900

        self.screenDims = numpy.array([self.length, self.length])
        self.screen = pygame.Surface((self.length, self.length))

        #self.screen = pygame.display.set_mode(self.screenDims)

        #if __debug__:
        #    self.screen = pygame.display.set_mode(self.screenDims)
        #    pygame.display.init()
        #    pygame.display.set_caption('A.I.gar')

    def drawCells(self, cells, fovPos, fovSize):
        for cell in cells:
            self.drawSingleCell(cell, fovPos, fovSize)


    def drawSingleCell(self, cell, fovPos, fovSize):
        screen = self.screen
        unscaledRad = cell.getRadius()
        unscaledPos = numpy.array(cell.getPos())
        color = cell.getColor()

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

    def drawAllCells(self, player):
        fovPos = player.getFovPos()
        fovSize = player.getFovSize()
        pellets = self.field.getPelletsInFov(fovPos, fovSize)
        blobs = self.field.getBlobsInFov(fovPos, fovSize)
        viruses = self.field.getVirusesInFov(fovPos, fovSize)
        playerCells = self.field.getPlayerCellsInFov(fovPos, fovSize)
        allCells = pellets + blobs + viruses + playerCells
        allCells.sort(key = lambda p: p.getMass())

        self.drawCells(allCells, fovPos, fovSize)
        



    def draw_cnnInput(self, player):
        self.screen.fill(WHITE)
        self.drawAllCells(player)
        #if __debug__:
            #if not self.parameters.CNN_P_RGB:
            #imgdata = pygame.surfarray.array3d(self.screen)
            #imgdata = self.grayscale_RGB(imgdata)
            #self.screen.blit(imgdata,(0,0))

            #pygame.display.update()

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

    def get_cnn_inputRGB(self, player):
        self.draw_cnnInput(player)
        imgdata = pygame.surfarray.array3d(self.screen)
        #if not self.parameters.CNN_P_RGB:
        #    imgdata = self.grayscale(imgdata)
        return imgdata

    def grayscale(self, arr):
        arr = numpy.average(arr, axis=2, weights=[0.298, 0.587, 0.114])
        shape = numpy.shape(arr)
        arr = arr.reshape(list(shape) + [1])
        return arr

    def grayscale_RGB(self, arr):
        arr = arr.dot([0.298, 0.587, 0.114])[:, :, None].repeat(3, axis=2)
        return pygame.surfarray.make_surface(arr)



