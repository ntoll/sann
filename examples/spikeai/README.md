# Lego Spike AI

Neural networks are also used in the real world - and this example shows how
to evolve an artificial neural network to then run on a 
[Lego Spike Prime](https://spike.legoeducation.com/prime/lobby/) configured
work as a motorised vehicle with a colour and distance sensor attached.

The resulting vehicle will avoid collisions and obey the following colour
coded "road signs":

* Red - stop
* Green - forward
* Blue - turn left
* Yellow - turn right
* No colour detected - same as green.

The distance sensor will give six possible outputs: 0 (nothing detected) and
1-5 (object detected with 1 the closest, and 5 the furthest away).

The neural network has an input layer of 11 nodes: five inputs indicating
each of the different colours that could be detected (red, green, blue,
yellow and none), and six inputs indicating the six possible distance
measurements (1-5, none). The network has four outputs used to control the
motors: left forward, left backward, right forward, right backward. If set to
below 0.2 the output represents stop. The highest forward/backward for each
motor is the one used to control the motor itself. For example, if the left
forward node had an output of 0.6 and the left backward node had an output
of 0.8 then the motor would run backwards.