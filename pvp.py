import pygame
import numpy as np
import random

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
FPS = 60

# Screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("1v1 Game with Smart AI")

# Clock
clock = pygame.time.Clock()

# Load images
player_image = pygame.image.load('c:/Users/Redspark/Downloads/gunp-removebg-preview.png')
player_image = pygame.transform.scale(player_image, (50, 50))
player2_image = pygame.transform.flip(player_image, True, False)

background_image = pygame.image.load('c:/Users/Redspark/Downloads/handwritten digits/bg path.PNG')
background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))

# Bullet class
class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y, dx, dy):
        super().__init__()
        self.image = pygame.Surface((5, 10))
        self.image.fill(RED)
        self.rect = self.image.get_rect(center=(x, y))
        self.dx = dx
        self.dy = dy
        self.speed = 10

    def update(self):
        self.rect.x += self.dx * self.speed
        self.rect.y += self.dy * self.speed
        if self.rect.bottom < 0 or self.rect.left > WIDTH or self.rect.right < 0 or self.rect.top > HEIGHT:
            self.kill()

# Player class
class Player(pygame.sprite.Sprite):
    def __init__(self, start_x, start_y, image, is_ai=False):
        super().__init__()
        self.image = image
        self.rect = self.image.get_rect(center=(start_x, start_y))
        self.speed = 5
        self.bullets = pygame.sprite.Group()
        self.last_shot_time = pygame.time.get_ticks()
        self.is_ai = is_ai
        self.health = 100

    def move(self, dx, dy):
        if not self.is_ai:
            self.rect.x += dx * self.speed
            self.rect.y += dy * self.speed
            self.rect.x = max(0, min(WIDTH - self.rect.width, self.rect.x))
            self.rect.y = max(0, min(HEIGHT - self.rect.height, self.rect.y))

    def shoot(self, target_x, target_y):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_shot_time > 500:  # Cooldown of 500ms
            dx = target_x - self.rect.centerx
            dy = target_y - self.rect.centery
            distance = np.sqrt(dx**2 + dy**2)
            dx, dy = dx / distance, dy / distance  # Normalize direction vector
            bullet = Bullet(self.rect.centerx, self.rect.centery, dx, dy)
            self.bullets.add(bullet)
            all_sprites.add(bullet)
            self.last_shot_time = current_time

    def update(self):
        self.bullets.update()

    def draw_health(self):
        pygame.draw.rect(screen, RED, (self.rect.x, self.rect.y - 20, self.health, 10))

# Create players
player1 = Player(WIDTH // 4, HEIGHT // 2 - 50, player_image, is_ai=True)
player2 = Player(3 * WIDTH // 4, HEIGHT // 2 - 50, player2_image, is_ai=True)
all_sprites = pygame.sprite.Group(player1, player2)

# RL Setup
gamma = 0.9  # Discount factor
alpha = 0.1  # Learning rate
epsilon = 0.1  # Exploration rate
num_actions = 5  # Number of possible actions

# Initialize Q-table
Q_table = np.zeros((WIDTH * HEIGHT, num_actions))

def get_state(player):
    return player.rect.y * WIDTH + player.rect.x

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, num_actions - 1)
    return np.argmax(Q_table[state])

def update_Q(state, action, reward, next_state):
    best_next_action = np.argmax(Q_table[next_state])
    Q_table[state, action] += alpha * (reward + gamma * Q_table[next_state, best_next_action] - Q_table[state, action])

def ai_move_and_dodge(player, opponent):
    # Move towards the center of the screen to avoid edges
    center_x = WIDTH // 2
    center_y = HEIGHT // 2
    dx = np.sign(center_x - player.rect.centerx)
    dy = np.sign(center_y - player.rect.centery)
    player.move(dx, dy)

    # If there are bullets, try to dodge
    for bullet in all_sprites:
        if isinstance(bullet, Bullet):
            if bullet.rect.colliderect(player.rect.inflate(50, 50)):  # Inflate player rect for collision check
                if bullet.rect.x < player.rect.x:
                    player.move(1, 0)  # Move right to dodge
                elif bullet.rect.x > player.rect.x:
                    player.move(-1, 0)  # Move left to dodge
                if bullet.rect.y < player.rect.y:
                    player.move(0, 1)  # Move down to dodge
                elif bullet.rect.y > player.rect.y:
                    player.move(0, -1)  # Move up to dodge
                break

def game_loop_rl():
    running = True
    while running:
        state1 = get_state(player1)
        state2 = get_state(player2)
        
        action1 = choose_action(state1)  # AI action for player 1
        action2 = choose_action(state2)  # AI action for player 2

        # Map actions to movements and shooting
        actions = {
            0: (-1, 0),  # Left
            1: (1, 0),   # Right
            2: (0, -1),  # Up
            3: (0, 1),   # Down
            4: (0, 0)    # Shoot (no movement)
        }

        dx1, dy1 = actions[action1]
        dx2, dy2 = actions[action2]

        if action1 != 4:
            player1.move(dx1, dy1)
        if action2 != 4:
            player2.move(dx2, dy2)

        if action1 == 4:
            player1.shoot(player2.rect.centerx, player2.rect.centery)
        if action2 == 4:
            player2.shoot(player1.rect.centerx, player1.rect.centery)

        ai_move_and_dodge(player2, player1)  # Player 2 AI behavior
        
        # Reward calculation and Q-table update
        reward1 = 0
        reward2 = 0
        
        if pygame.sprite.spritecollideany(player1, player2.bullets):
            reward1 = -10
            reward2 = 10
            player1.health -= 10
        
        if pygame.sprite.spritecollideany(player2, player1.bullets):
            reward2 = -10
            reward1 = 10
            player2.health -= 10

        next_state1 = get_state(player1)
        next_state2 = get_state(player2)

        update_Q(state1, action1, reward1, next_state1)
        update_Q(state2, action2, reward2, next_state2)

        state1 = next_state1
        state2 = next_state2

        # Check for end condition
        if player1.health <= 0 or player2.health <= 0:
            running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        all_sprites.update()

        screen.fill(WHITE)
        screen.blit(background_image, (0, 0))  # Draw the background image
        all_sprites.draw(screen)
        player1.draw_health()
        player2.draw_health()
        pygame.display.flip()
        clock.tick(FPS)

    winner = "Player 2" if player1.health <= 0 else "Player 1"
    print(f"{winner} wins!")
    pygame.quit()

if __name__ == "__main__":
    game_loop_rl()
