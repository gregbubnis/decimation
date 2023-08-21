







#from decimate.decimate import incremental_decimation

from decimate.decimate import incremental_decimation, make_trimesh, pimp_meshes, twisty_viz, dump_mp4
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":

    dec_steps = 50
    
    viz_opts = dict(
            colormap=plt.cm.winter,
            #bg_color=[20, 20, 30],
            bg_color=[255, 255, 255],
    )

    # input_filename = 'topodata/data_halfdomecrop_srtm30m_0030m.csv'
    # output_filename = 'movie-halfdome.mp4'
    # azimuth, tilt, dec_min = 240, 15, 100

    # input_filename = 'topodata/data_mammoth_srtm30m_0050m.csv'
    # output_filename = 'movie-mammoth.mp4'
    # azimuth, tilt, dec_min = 180, 15, 200

    input_filename = 'topodata/data_morrison_srtm30m_0050m.csv'
    output_filename = 'movie-morrison.mp4'
    azimuth, tilt, dec_min = 90, 30, 300

    # input_filename = 'topodata/data_dana_srtm30m_0040m.csv'
    # output_filename = 'movie-mtdana.mp4'
    # azimuth, tilt, dec_min = 180, 20, 200

    # input_filename = 'topodata/data_cannon_srtm30m_0040m.csv'
    # output_filename = 'movie-cannon.mp4'
    # azimuth, tilt, dec_min = 115, 15, 200

    # input_filename = 'topodata/data_tom2emerson_srtm30m_0100m.csv'
    # output_filename = 'movie-tom2emerson.mp4'
    # azimuth, tilt, dec_min = 90, 15, 400

    # input_filename = 'topodata/data_schralpine_srtm30m_0040m.csv'
    # output_filename = 'movie-schralpine.mp4'
    # azimuth, tilt, dec_min = 180, 15, 250

    # input_filename = '../topodata/data_emeraldbay_srtm30m_0040m.csv'
    # output_filename = 'movie-emerald.mp4'
    # azimuth, tilt, dec_min = 90, 5, 1500




    # # set up output folder
    # dest = 'test-decimate-%s' % (get_datetime(taglen=4))
    # os.makedirs(dest)

    # # download halfdome data at 30m resolution
    # corners = [(37.7500, -119.5430), (37.7390, -119.5280)]
    # data = get_area(corners=corners, stride=30)
    # data.to_csv('data_halfdomecrop_srtm30m_0030m.csv', float_format='%8.4f')

    # # can load from file if surface data was already downloaded
    data = pd.read_csv(input_filename, index_col=0)


    viz_opts['azimuth'] = azimuth
    viz_opts['tilt'] = tilt


    mesh = make_trimesh(data[['x', 'y', 'elevation']].values)
    meshes = incremental_decimation(mesh, dec_min, dec_steps)
    pimped = pimp_meshes(meshes, **viz_opts)
    mov = twisty_viz(pimped, **viz_opts)
    if mov is not None:
        dump_mp4(mov, output_filename=output_filename)

