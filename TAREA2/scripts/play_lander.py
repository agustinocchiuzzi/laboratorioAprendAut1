"""Play LunarLander in a local window, without running the notebook or training."""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--check-window', action='store_true',
                        help='Open the real game window for two seconds, then exit.')
    args = parser.parse_args()

    # A previous headless notebook session can leave SDL configured without a window.
    if sys.platform == 'darwin' and os.environ.get('SDL_VIDEODRIVER') in {'dummy', 'offscreen'}:
        os.environ.pop('SDL_VIDEODRIVER')
    try:
        import gymnasium as gym
        import pygame
    except ImportError as error:
        parser.exit(1, f'{error}\nRun this script using the project .venv/bin/python.\n')

    env = gym.make('LunarLander-v3', render_mode='human')
    try:
        env.reset(seed=args.seed)
        pygame.display.set_caption('LunarLander — W / A / D to fly; Q or Esc to quit')
        driver = pygame.display.get_driver()
        if driver in {'dummy', 'offscreen'}:
            raise RuntimeError('No visible display. Run this script in a local desktop terminal.')
        print(f'Window open (SDL: {driver}). Click it, then use W/↑, A/←, D/→. Q/Esc quits.', flush=True)
        clock = pygame.time.Clock()
        started = pygame.time.get_ticks()
        episode_return = 0.0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE)
                ):
                    return
            if args.check_window and pygame.time.get_ticks() - started >= 2000:
                print('Visible-window check passed.', flush=True)
                return
            keys = pygame.key.get_pressed()
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                action = 3
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                action = 1
            elif keys[pygame.K_w] or keys[pygame.K_UP]:
                action = 2
            else:
                action = 0
            _, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            if terminated or truncated:
                print(f'Episode return: {episode_return:.1f}. Starting a new attempt.', flush=True)
                episode_return = 0.0
                env.reset()
            clock.tick(50)
    finally:
        env.close()
        pygame.quit()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
