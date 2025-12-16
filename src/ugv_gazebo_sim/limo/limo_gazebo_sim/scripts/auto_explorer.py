#!/usr/bin/env python3
import rospy
import actionlib
import random
import numpy as np
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import OccupancyGrid

class Explorer:
    def __init__(self):
        rospy.init_node('auto_explorer')
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base...")
        self.client.wait_for_server()
        self.map_data = None
        rospy.Subscriber('/map', OccupancyGrid, self.map_cb)

    def map_cb(self, msg):
        self.map_data = msg

    def get_valid_goal(self):
        if not self.map_data:
            return None

        width = self.map_data.info.width
        height = self.map_data.info.height
        grid = np.array(self.map_data.data).reshape(height, width)

        # Find free cells
        free_cells = np.argwhere(grid == 0)

        # Keep only cells with safe margin from obstacles
        safe_cells = []
        margin = 2  # cells around should be free
        for r, c in free_cells:
            r_min = max(r - margin, 0)
            r_max = min(r + margin + 1, height)
            c_min = max(c - margin, 0)
            c_max = min(c + margin + 1, width)
            neighborhood = grid[r_min:r_max, c_min:c_max]
            if np.all(neighborhood == 0):
                safe_cells.append((r, c))

        if not safe_cells:
            return None

        r, c = random.choice(safe_cells)
        x = c * self.map_data.info.resolution + self.map_data.info.origin.position.x
        y = r * self.map_data.info.resolution + self.map_data.info.origin.position.y
        return x, y

    def run(self):
        rospy.sleep(2)  # wait for move_base to start
        rate = rospy.Rate(0.2)  # pick new goal at most every 5 sec
        while not rospy.is_shutdown():
            target = self.get_valid_goal()
            if target:
                goal = MoveBaseGoal()
                goal.target_pose.header.frame_id = "map"
                goal.target_pose.header.stamp = rospy.Time.now()
                goal.target_pose.pose.position.x = target[0]
                goal.target_pose.pose.position.y = target[1]
                goal.target_pose.pose.orientation.w = 1.0
                rospy.loginfo(f"Going to: {target}")
                self.client.send_goal(goal)
                # Wait until goal reached or timeout
                self.client.wait_for_result(rospy.Duration(30.0))
            rate.sleep()

if __name__ == '__main__':
    Explorer().run()
