

# Decimation
Chunky visualization of mountainous terrain.
![mountains surrounding Emerald Bay, Lake Tahoe](media/emerald_bay_header.png)

# Introduction
Nowadays, high resolution terrain maps are widely and freely available from tools like Google Earth and popular GPS apps. Here, rather than trying to maximize accuracy and realism, the aim is to create chunky terrain visualizations that look cool while still preserving accuracy to the extent possible.

In computer graphics, schemes to simplify complex mesh surfaces are important for reducing unnecessary processing time. One well known method is the quadric decimation algorithm devised by Garland and Heckbert in 1997. Shown below are two of their examples, one of a cow and one of a terrain map of Crater Lake. This method merges vertices adaptively so that flat, unfeatured areas (cow torso, lake surface) comprise a few large faces and shaped features (horns, crater rim) remain well resolved with smaller faces. 

![alt text](media/cows.png)
![alt text](media/crater_lake.png)

This is an exploratory project. So far, the goal has been to create visualizations of my favorite mountain landscapes. The workflow constists of fetching terrain data from a public API, applying this decimation algorithm and then visualizing the results.

# Setup
Conda env, pip install

# Usage
Jupyter notebook

<!---
[1] https://www.geom.at/terrain-triangulation/
[2] https://maps3d.io/
[3] https://github.com/kk7ds/gaiagpsclient
[4] https://code.wsl.ch/snow-models/snowpack
[5] https://snowpack.slf.ch/

# things learned
- some elevation datasets are wildly inaccurate (check out the matterhorn!)
--->