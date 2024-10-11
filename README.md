# GeneralGRRayTracer

This is a 'demo' for a generalized raycaster in asymptomatically flat spacetimes. The final goal of this project, which is not yet attained, is writing a fully coordinate independent raycasting suite that can operate in _any_ asymptomatically flat spacetime that has a trivial interior and boundary (i.e. no direct support for wormhole visualizations for now). Performance remains secondary for now, but since we are using SciML suites in the 'hot' part of the code, future optimizations should be fairly trivial. 

The current main problem in the project remains the (automatically and globally infered) coordinate transformation into the cannonical Minkowski coordinates in the assymptically flat region (which remains problematic due to the fact that while it is possible to find _a_ local transformation easily, a global transformation is much harder to find due to rotational freedom of the Minkowski coordinates).

The raycaster has limited ability of rendering analytic surfaces thru the SciML event handling suite, however, this might impact GPU ensemble performance. 

## Examples

Semi-transparent accreation disc of a Schwarschild blackhole:

![sch_test](https://github.com/ArchHem/GeneralGRRayTracer/blob/main/renders/test7.png)




