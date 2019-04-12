from view.view import *
import pygame
from keras.utils import plot_model


class Controller:
    """
    Initializing the 'root' main container, the model, the view,
    """

    def __init__(self, model, viewEnabled, view, mouseEnabled):
        self.model = model
        self.view = view
        self.screenWidth, self.screenHeight = self.view.getScreenDims()
        self.running = True
        self.viewEnabled = viewEnabled
        self.mouseEnabled = mouseEnabled
        self.selectedPlayer = None
        self.paused = False

    def process_input(self):
        humanList = self.model.getHumans()
        humanCommandPoint = []
        if self.model.hasHuman():
            for human in humanList:
                humanCommandPoint.append([self.screenWidth/2, self.screenHeight/2])
                if human.getIsAlive():
                    human.setSplit(False)
                    human.setEject(False)
            if self.mouseEnabled:
                # Human1 direction control
                humanCommandPoint[0] = pygame.mouse.get_pos()

                #Human2 direction control
                if len(humanList) > 1:
                    keys = pygame.key.get_pressed()
                    if humanList[1].getIsAlive():
                        if keys[pygame.K_UP]:
                            humanCommandPoint[1][1] -= self.screenHeight/2
                        if keys[pygame.K_DOWN]:
                            humanCommandPoint[1][1] += self.screenHeight/2
                        if keys[pygame.K_LEFT]:
                            humanCommandPoint[1][0] -= self.screenWidth/2
                        if keys[pygame.K_RIGHT]:
                            humanCommandPoint[1][0] += self.screenWidth/2
                    #Human3 direction controls
                    if len(humanList) > 2 and humanList[2].getIsAlive():
                        if keys[pygame.K_w]:
                            humanCommandPoint[2][1] -= self.screenHeight/2
                        if keys[pygame.K_s]:
                            humanCommandPoint[2][1] += self.screenHeight/2
                        if keys[pygame.K_a]:
                            humanCommandPoint[2][0] -= self.screenWidth/2
                        if keys[pygame.K_d]:
                            humanCommandPoint[2][0] += self.screenWidth/2
            else:
                # Human1 direction control

                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]:
                    humanCommandPoint[0][1] -= self.screenHeight / 2
                if keys[pygame.K_DOWN]:
                    humanCommandPoint[0][1] += self.screenHeight / 2
                if keys[pygame.K_LEFT]:
                    humanCommandPoint[0][0] -= self.screenWidth / 2
                if keys[pygame.K_RIGHT]:
                    humanCommandPoint[0][0] += self.screenWidth / 2

                 # Human2 direction control
                if len(humanList) > 1 and humanList[1].getIsAlive():
                    if keys[pygame.K_w]:
                        humanCommandPoint[1][1] -= self.screenHeight/2
                    if keys[pygame.K_s]:
                        humanCommandPoint[1][1] += self.screenHeight/2
                    if keys[pygame.K_a]:
                        humanCommandPoint[1][0] -= self.screenWidth/2
                    if keys[pygame.K_d]:
                        humanCommandPoint[1][0] += self.screenWidth/2


            for i in range(len(humanList)):
                self.mousePosition(humanList[i], humanCommandPoint[i], i)

        for event in pygame.event.get():
            # Event types
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                # "Escape" to Quit
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                # Get controls for humans
                if humanList:
                    #Human1 controls
                    human1 = humanList[0]
                    if human1.getIsAlive():
                        # "space" to Split
                        if event.key == pygame.K_SPACE and human1.getCanSplit():
                            human1.setSplit(True)
                        # "w" to Eject
                        elif event.key == pygame.K_b and human1.getCanEject():
                            human1.setEject(True)
                        elif event.key == pygame.K_m:
                            human1.addMass(human1.getTotalMass() * 0.2)
                    if len(humanList) > 1:
                        #Human2 controls
                        human2 = humanList[1]
                        if human2.getIsAlive():
                            # "." to Split
                            if event.key == pygame.K_k and human2.getCanSplit():
                                human2.setSplit(True)
                            # "-" to Eject
                            elif event.key == pygame.K_l and human2.getCanEject():
                                human2.setEject(True)
                            elif event.key == pygame.K_j:
                                human2.addMass(human2.getTotalMass() * 0.2)
                            humanList[1] = human2
                    if len(humanList) > 2:
                        #Human3 controls
                        human3 = humanList[2]
                        if human3.getIsAlive():
                            # "e" to Split
                            if event.key == pygame.K_e and human3.getCanSplit():
                                human3.setSplit(True)
                            # "r" to Eject
                            elif event.key == pygame.K_q and human3.getCanEject():
                                human3.setEject(True)
                            elif event.key == pygame.K_r:
                                human3.addMass(human3.getTotalMass() * 0.2)

                # Switch between spectated players if single player is spectated
                if self.model.hasPlayerSpectator() and not humanList:
                    spectatedPlayer = self.model.getSpectatedPlayer()
                    if event.key == pygame.K_RIGHT:
                        players = self.model.getPlayers()
                        nextPlayerIndex = (players.index(spectatedPlayer) + 1) % len(players)
                        nextPlayer = players[nextPlayerIndex]
                        self.model.setSpectatedPlayer(nextPlayer)
                    if event.key == pygame.K_LEFT:
                        players = self.model.getPlayers()
                        nextPlayerIndex = (players.index(spectatedPlayer) - 1) % len(players)
                        nextPlayer = players[nextPlayerIndex]
                        self.model.setSpectatedPlayer(nextPlayer)

                if not humanList:
                    # Plot mean td errors and mean rewards
                    if event.key == pygame.K_t:
                        for bot in self.model.bots:
                            if bot.getType() == "NN" and bot.learningAlg is not None:
                                print("Before")
                                print(bot.learningAlg.network.valueNetwork.get_weights()[0])
                                bot.learningAlg.reset_weights()
                                print("After")
                                print( bot.learningAlg.network.valueNetwork.get_weights()[0])

                    if event.key == pygame.K_n:
                        for bot in self.model.bots:
                            if bot.getType() == "NN" and bot.learningAlg is not None:
                                model = bot.getLearningAlg().network.valueNetwork
                                break
                        plot_model(model, to_file='model.png', show_shapes=True)

                    if event.key == pygame.K_p:
                        self.model.save()

            # Handle player selection by clicking and view dis/enabling
            if not humanList:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.selectPlayer()
                if event.type == pygame.MOUSEBUTTONUP:
                    if self.viewEnabled and not self.selectedPlayer:
                        self.setViewEnabled(False)
                    elif not self.viewEnabled:
                        self.setViewEnabled(True)

    # Sets a player to selected if the mouse is on them
    def selectPlayer(self):
        fovPos = self.model.getFovPos(None)
        fovSize = self.model.getFovSize(None)
        relativeMousePos = self.view.viewToModelScaling(pygame.mouse.get_pos(), fovPos, fovSize)
        if self.selectedPlayer:
            self.selectedPlayer.setSelected(False)
            self.selectedPlayer = None
        for cell in self.model.getPlayerCells():
            radius = cell.getRadius()
            if cell.squareDist(cell.getPos(), relativeMousePos) < radius * radius:
                self.selectedPlayer = cell.getPlayer()
                self.selectedPlayer.setSelected(True)
                break

    def setViewEnabled(self, val):
        self.viewEnabled = val
        self.model.setViewEnabled(val)

    # Find the point where the player moved, taking into account that he only sees the fov
    def mousePosition(self, human, mousePos, humanNr):
        fovPos = self.model.getFovPos(humanNr)
        fovSize = self.model.getFovSize(humanNr)
        relativeMousePos = self.view.viewToModelScaling(mousePos, fovPos, fovSize)
        human.setMoveTowards(relativeMousePos)
