# Optimization-of-Image-Diffraction

In this project we designed and simulated a phase mask that can be used to clasify hand-written numbers (based on MNist dataset).
We decided to that the classification will be done by projecting the picture on the mask, and letting it propagate through free space until it hits a sensor that measures the intensity of the light field. The sensor is Divided to 4, when the intensity is maximal at a certain quarter - it interprets it as a certain number (0, 1, 2 or 3).

In order to make the mask such that light of a number image difracting through it will result with maximum intensity in the correct quarter, we chose to use an optimization processes that is based on the genetic optimization algorithm. Starting with a group of complete random masks, in each epoch the best half of masks were cross-bred such that their properties are conserved in their "children". In each epoch, each of the children got also a kick of a "mutation" - a random change in a small portion of its properties, that is used to deal better with convergence to a local sub-optimal solution. 
