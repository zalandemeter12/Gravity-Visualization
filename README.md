# Gravity-Visualization
## Specification
A rubber sheet simulator demonstrating gravity. Our rubber sheet with a flat torus topology (what goes out comes in on the opposite side) is initially viewed from above, on which we can place large, non-moving bodies by pressing the right mouse button, and small balls can be slid frictionlessly from the bottom left corner by pressing the left mouse button, when the position of the press, together with the bottom left corner, gives the initial velocity. The high mass bodies at rest curve the space, i.e. deform the rubber sheet, but they are not visible. The indentation caused is m/(r + r0) at a distance r from the centre of the mass, where r0 is half a percent of the width of the rubber sheet and m is the mass that increases as the bodies are picked up one after the other. The rubber sheet is optically lumpy, with a diffuse and ambient coefficient that darkens in steps according to the indentation. The spheres are coloured diffuse-specular, with negligible field curvature and size. Pressing SPACE causes our virtual camera to stick to the first ball not yet absorbed, so we can follow its point of view. The balls colliding with the masses are absorbed, the collisions between the balls need not be considered. The rubber sheet is illuminated by two point light sources rotating around each other's initial position according to the following quaternion (t is time):

q=[cos(t/4), sin(t/4)cos(t)/2, sin(t/4)sin(t)/2, sin(t/4)√(3/4])

## Results
![Gravity-Visualization](https://i.imgur.com/Cv1Bnig.png)



