# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

# x = np.arange(0,4*np.pi,0.1)
# y = np.sin(x)

# plt.plot(x,y)
# plt.title('Sin Component Curve')
# plt.xlabel('Angle in Radians')
# plt.ylabel('Output Value')
# plt.show()

# x = np.arange(0,4*np.pi,0.1)
# y = (x*180/np.pi)
# y = np.where(y>360,y-360,y)
# y = y-180

# plt.plot(x,y)
# plt.title('Traditional Angle Curve')
# plt.xlabel('Angle in Radians')
# plt.ylabel('Output Value')
# plt.yticks(np.arange(-180,181,step=45))
# plt.show()

x = 1
y = 1
y = np.arctan2(y,x)
plt.plot(x,y)

x = 2
y = 1
y = np.arctan2(y,x)
plt.plot(x,y)

x = 1
y = 2
y = np.arctan2(y,x)
plt.plot(x,y)


plt.title('Arctan2 Curve')
plt.xlabel('Angle in Radians')
plt.ylabel('Output Value')
plt.show()
