import pygame
import numpy as np

eps = 50
min_pts = 4
cluster_colors = [(0, 255, 0), (255, 255, 0), (255, 0, 0)]
pygame.init()
screen = pygame.display.set_mode((800, 600))
screen.fill((255, 255, 255))
clock = pygame.time.Clock()
drawing = False
clustering = False
points = []
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if not clustering:
                drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                clustering = True

    if drawing:
        mouse_pos = pygame.mouse.get_pos()
        points.append(mouse_pos)
        pygame.draw.circle(screen, (0, 0, 0), mouse_pos, 10)

    if clustering:
        distances = np.linalg.norm(np.array(points)[:, np.newaxis] - np.array(points), axis=2)

        core_points = []
        for i in range(len(points)):
            if np.sum(distances[i] <= eps) >= min_pts:
                core_points.append(i)

        clusters = []
        visited = np.zeros(len(points), dtype=bool)
        for core_point in core_points:
            if not visited[core_point]:
                cluster = [core_point]
                visited[core_point] = True
                queue = [core_point]
                while queue:
                    current_point = queue.pop(0)
                    neighbors = np.where(distances[current_point] <= eps)[0]
                    for neighbor in neighbors:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                            cluster.append(neighbor)
                clusters.append(cluster)

        for i, cluster in enumerate(clusters):
            for point_idx in cluster:
                pygame.draw.circle(screen, cluster_colors[i % len(cluster_colors)], points[point_idx], 10)

    pygame.display.update()
    clock.tick(60)

pygame.quit()