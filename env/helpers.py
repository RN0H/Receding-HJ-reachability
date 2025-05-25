import numpy as np


def generate_obstacles(num_obstacles, agent_center, tolerance_u, goal, max_attempts=100):
    """
    Generating random circular obstacles that do not interfere with the goal and the agent
    """
    obstacles = []
    attempts = 0

    while len(obstacles) < num_obstacles and attempts < max_attempts:
        
        # Optionally adjust sampling region or bias if too many attempts
        if attempts > 0 and attempts % 100 == 0:
            print(f"[Info] Attempt #{attempts}, obstacles placed: {len(obstacles)}")

        x, y = np.random.uniform(0, ENV_SIZE, size=2)
        r = np.random.uniform(0.3, 1.0)

        # Check overlap with unicycle and goal
        dist_to_agent = np.hypot(x - agent_center[0], y - agent_center[1])
        dist_to_goal = np.hypot(x - goal[0], y - goal[1])

        if dist_to_agent < r + 10 * tolerance_u or dist_to_goal < r + goal[-1] + 0.2:
            attempts += 1
            continue

        # Check for overlap with existing obstacles
        overlaps = False
        for ob in obstacles:
            dist = np.hypot(x - ob.x, y - ob.y)
            if dist < r + ob.r + 0.2:  # Add buffer
                overlaps = True
                break

        # Create New Obstacle
        if not overlaps:
            obstacles.append(Obstacle(x, y, r))
        attempts += 1

    if attempts == max_attempts:
        print("[Warning] Max attempts reached. Could not place all obstacles without overlap.")

    return obstacles
