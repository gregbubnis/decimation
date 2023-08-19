

# Decimation
Visualizing mountainous terrain with a chunky, minimalist aesthetic.

![mountains surrounding Emerald Bay, Lake Tahoe](media/emerald_bay_header.png)

# Introduction
Nowadays, high resolution terrain maps are widely and freely available from tools like Google Earth and popular GPS apps. Here, the goal is not maximum accuracy or realism, but rather to create chunky terrain visualizations that look cool and preserve accuracy to the extent possible.

In computer graphics, schemes to simplify complex mesh surfaces are important for reducing unnecessary processing time. A prominent example of such a scheme is the quadric decimation algorithm devised by Garland and Heckbert in 1997. Shown below are two examples from their seminal paper [2] on this method, one for a cow and one for a terrain map of Crater Lake. 
![alt text](media/cows.png)
![alt text](media/crater_lake.png)

This project constists of fetching terrain data from a public API, applying this decimation algorithm and then visualizing the results.

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