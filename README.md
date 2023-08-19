

# Decimation
A mini project for visualizing mountainous terrain with a chunky, minimalist aesthetic.

![mountains surrounding Emerald Bay, Lake Tahoe](media/emerald_bay_header.png)

# Introduction
Nowadays, high resolution terrain maps are widely and freely available from tools like Google Earth and popular GPS apps. One webapp, TouchTerrain [1], enables users to select a region on a map and generate a 3D printable terrain model. Here, my goal is not maximum accuracy, but rather to create chunky terrain visualizations that preserve accuracy to the extent possible.

Why? For me, it is sufficient that it looks cool.

For the computer graphics community, "decimation" schemes are very important for simplifying mesh surfaces and reducing unnecessary processing time. The 25 year old quadric decimation algorithm devised by Garland and Heckbert is just the tool that we need. Shown below are two examples from their seminal paper [2] on this method, one for a cow mesh and one for a terrain map of Crater Lake. 
![alt text](media/cows.png)
![alt text](media/crater_lake.png)


[1] https://touchterrain.geol.iastate.edu/  
[2] Garland and Heckbert, Surface Simplification Using Quadric Error Metrics, Proceedings of the 24th annual conference on Computer graphics and interactive techniques (1997)

<!---
Here, the goal is to convert high density terrain datasets into very low density meshes while still optimizing for accuand modify it very low density mesh that is still optimized for accuracy. Visually this results in large 

Here the goal is also to maximize accuracy, but while using sparse rather than dense elevation data. This concept is familiar to the 3D graphics and geoscience communities (and probably others), and it could be called downsampling, coarse-graining, or decimation.

I think that it looks pleasing, and decided to give it a try on some of my favorite mountains! The workflow is to get hands on some high resolution terrain data, test schemes to winnow it down, and visualize the results.

# Uniform downsampling


# Greedy Mesh: quick and dirty
One way to 
Visual comparison of uniform downsampling 



=======================================

# about
This is a toy project that incorporates three of my seemingly disjoint interests: mountains, optimization problems and wireframe visualization.

This winter, (22/23) I spent a lot of time on backcountry skis in different parts of California's Sierra Nevada, seeing the striking terrain and topography first hand. Mountainous landscapes can be depicted in myriad ways by artists, photographers, sculptors, map-makers, authors and so on. Among these, I like the coarse, wire-frame depictions that some resorts and clothing companies use and wanted to be able to make my own.

To me, the aesthetic appeal is that they can capture the essential, and recognizable features of a mountain landscape while being stylisticaly simple. The extra appeal is that a neat mathematical problem is lurking here; Building an accurate coarse mesh (starting from high density mesh data) turns out to be a non-trivial optimization problem!

So this project is an exploration of how to create these coarse models for the very mountains that I explored!


High density 3D meshes (2.5D, but more on that later) can accurately, and realistically model mountain topography.



I spent much of the 22/23 winter (and spring!) backcountry skiing in California's Sierra Nevada, seeing the striking terrain and topography first hand. I have always liked the blocky, abstract depictions of mountain landscapes, especially the ones that use relatively few points and faces to reproduce the topography as accurately as possible.





- mammoth
- basin
- tallac
- hulk
- wahoo
- scheelite
- morrison (baldwin!)
- mammoth crest?
- humphreys-basin-tom!
- alpine!



[1] https://www.geom.at/terrain-triangulation/
[2] https://maps3d.io/
[3] https://github.com/kk7ds/gaiagpsclient
[4] https://code.wsl.ch/snow-models/snowpack
[5] https://snowpack.slf.ch/

# things learned
- some elevation datasets are hugely flawed (check out the matterhorn!)
- for a triangle mesh surface, the order of vertices determines the direction of face normals, i.e. (1,2,3) and (2,1,3) will point opposite directions, and this is used by the shader!
- just remembered some random discussions with christian blau. We agreed that density of states is cleaner and preferable to probability, kBT is the real unit of energy, and that it might just be probability the whole way down (not energy)
- using thousands of individual triangles to render delaunay triangles will bog down the GUI, so better to make long contiguous lines that trace the edges.  
- dataframe indexing by integer (iloc) is faster than by index (loc)




--->