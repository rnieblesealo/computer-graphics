#!/home/rniebles/Developer/GRAPHICS/.cgenv/bin/python3

import random
import pygame

pygame.init()
pygame.display.set_caption(title="Assignment 1: Rafael Niebles")

screen = pygame.display.set_mode((500, 500))
clock = pygame.time.Clock()
running = True

while running:
    # need quit mechanism
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # sent by window x click i think
            running = False

    # randomize color every frame
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    a = 255

    screen.fill((r, g, b, a))  # fill screen with color
    pygame.display.flip()  # swap buffer
    clock.tick(30)  # limit to 30fps

pygame.quit()
