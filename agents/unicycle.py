import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from IPython.display import HTML


# Unicycle model parameters
class Unicycle:
    def __init__(self, x=0.0, y=0.0, theta=0.0, length=0.4, width=0.2):
        """
        initialize unicycle state and geometry
        """
        self.length = length
        self.width = width
        self.set_state(x, y, theta)

    
    def set_state(self, x, y, theta):
        """
        Updating the state of the unicycle car

        """
        self.x = x
        self.y = y
        self.theta = theta  # heading angle in radians
        
        # components of along x and y
        self.dx = self.length * np.cos(self.theta)
        self.dy = self.length * np.sin(self.theta)
        self.center = ((self.x + self.dx) / 2, (self.y + self.dy) / 2)
        self.build_geometry()
        

    def build_geometry(self):
        """
        Building the geometry for the car. In this case a triangle
        """
        # finding the rear coordinates of the triangle
        left = (self.x - 0.5 * self.width * np.sin(self.theta), self.y + 0.5 * self.width * np.cos(self.theta))
        right = (self.x + 0.5 * self.width * np.sin(self.theta), self.y - 0.5 * self.width * np.cos(self.theta))

        # tip coorindate is the origin + the components
        tip = (self.center[0] * 2.0, self.center[1] * 2.0)

        # last coordinate has to match the first one in order to  complete the outline
        self.geometry = np.array([left, right, tip, left])


    def draw_patch(self, ax):
        """
        plotting the geometry
        """
        self.patch = ax.fill(self.geometry[:, 0], self.geometry[:, 1], color='orange')[0]
        return self.patch

    def update_patch(self):            
        """
        updating the geometry
        """
        self.patch.set_xy(self.geometry)
        return [self.patch]
    


if __name__ == "__main__":
    ##### Test ######
    unicycle = Unicycle(x=2.0, y=3.0, theta=np.pi/4)