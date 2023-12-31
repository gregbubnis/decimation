#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import itertools
import open3d
import subprocess as sp
import shlex
import copy
import scipy

def set_rotation(g, azimuth=0, tilt=0):
    """rotates a Trimesh around its bounding box center. In addition to setting
    azimuth and tilt, the mesh is rotated so that elevation (data z-axis) is
    oriented up-down (display y-axis).
    
    Args:
        g (Trimesh): Open3D Trimesh
        azimuth (float): viewer azimuth (0=N, 90=W, 180=S, 270=E)
        tilt (float): tilt the mesh towards viewer
    Returns:
    """
    ctr = g.get_axis_aligned_bounding_box().get_center()
    r1 = open3d.geometry.get_rotation_matrix_from_zyx(np.array([0 , -azimuth*np.pi/180., -np.pi/2]))
    r2 = open3d.geometry.get_rotation_matrix_from_zyx(np.array([0, 0, tilt*np.pi/180.]))
    g.rotate(r1, center=ctr).rotate(r2, center=ctr)

def dump_mp4(mov, output_filename='movie.mp4', fps=60, crf=18):
    """dump frames to mp4

    adapted from https://stackoverflow.com/a/61281547/6474403

    Args:
        mov (ndarray): array of movie frames (num_frames, height, width, 3) uint8
        output_filename (str): output filename (will overwrite)
        fps (float): frame per second
        crf (float): constant rate factor, 0-51 with 0 being lossless, and 51 being worst
    """
    if mov is None:
        return

    width, height, n_frames= mov.shape[2], mov.shape[1], len(mov)

    # Open ffmpeg application as sub-process
    # FFmpeg input PIPE: RAW images in BGR color format
    # FFmpeg output MP4 file encoded with HEVC codec.
    # Arguments list:
    # -y                   Overwrite output file without asking
    # -s {width}x{height}  Input resolution width x height (1344x756)
    # -pixel_format bgr24  Input frame color format is BGR with 8 bits per color component
    # -f rawvideo          Input format: raw video
    # -r {fps}             Frame rate: fps (25fps)
    # -i pipe:             ffmpeg input is a PIPE
    # -vcodec libx265      Video codec: H.265 (HEVC)
    # -pix_fmt yuv420p     Output video color space YUV420 (saving space compared to YUV444)
    # -crf 24              Constant quality encoding (lower value for higher quality and larger output file).
    # {output_filename}    Output file name: output_filename (output.mp4)
    process = sp.Popen(shlex.split(f'ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx265 -pix_fmt yuv420p -crf {crf} {output_filename}'), stdin=sp.PIPE)

    # Build synthetic video frames and write them to ffmpeg input stream.
    for i in range(n_frames):
        # Build synthetic image for testing ("render" a video frame).
        # img = np.full((height, width, 3), 60, np.uint8)
        # cv2.putText(img, str(i+1), (width//2-100*len(str(i+1)), height//2+100), cv2.FONT_HERSHEY_DUPLEX, 10, (255, 30, 30), 20)  # Blue number
        #img = cv2.cvtColor(mov[i], cv2.COLOR_RGB2BGR)
        img = mov[i][:, :, [2,1,0]]

        # Write raw video frame to input stream of ffmpeg sub-process.
        process.stdin.write(img.tobytes())

    # Close and flush stdin
    process.stdin.close()

    # Wait for sub-process to finish
    process.wait()

    # Terminate the sub-process
    process.terminate()  # Note: We don't have to terminate the sub-process (after process.wait(), the sub-process is supposed to be closed).

def add_bbox_points(mesh):
    """Add bounding box corners as vertices of a Trimesh. This prevents
    translation drift due to re-centering after decimation. The added
    vertices are not part of any triangles so they are preserved during
    decimation.
    """
    c = copy.deepcopy(mesh)
    xyz = np.vstack((np.array(c.vertices), np.array(c.get_axis_aligned_bounding_box().get_box_points())))
    c.vertices = open3d.utility.Vector3dVector(xyz)
    return c

def pimp_meshes(meshes, **kwargs):
    """Staging a list of meshes for visualization. Vertices are colored, and
    meshes are re-oriented and re-aligned.

    Args:
        meshes (list): list of TriMesh objects
        colormap (): Matplotlib colormap or an RGB integer 3-tuple (0-255)
        azimuth (float): initial orientation (see `set_rotation()`)
        tilt (float): initial orientation (see `set_rotation()`)
    Returns:
        meshes (list): new list of TriMesh objects (input are unchanged)
    """
    defaults = dict(
        colormap=plt.cm.winter,
        azimuth=0,
        tilt=0,
    )
    colormap = kwargs.get('colormap', defaults['colormap'])
    azimuth = kwargs.get('azimuth', defaults['azimuth'])
    tilt = kwargs.get('tilt', defaults['tilt'])

    meshes = [copy.deepcopy(x) for x in meshes]
    # set vertex colors based on elevation
    for x in meshes:
        zvec = np.array(x.vertices)[:, 2]
        zvec = (zvec-np.min(zvec))/(np.max(zvec)-np.min(zvec))
        if isinstance(colormap, matplotlib.colors.LinearSegmentedColormap):
            x.vertex_colors = open3d.utility.Vector3dVector(colormap(zvec)[:, 0:3])
        elif isinstance(colormap, list):
            x.paint_uniform_color(np.asarray(colormap)/255)
        x.compute_vertex_normals() # so lighting works
    # Set initial rotation and align the bbox ctrs
    ref = np.array(meshes[0].get_axis_aligned_bounding_box().get_center())
    for x in meshes:
        set_rotation(x, azimuth, tilt)
        x.translate(ref-np.array(x.get_axis_aligned_bounding_box().get_center()))

    return meshes

def twisty_viz(meshes, **kwargs):
    """Mesh surface visualization that cycles through a list of geometries
    while twisting back and forth.

    The UI allows lots of fiddling (changing view options, zoom, resize, etc.)
    and pressing `h` will print a list of keyboard shortcuts. Extra options
    added here include
    - press spacebar to pause/resume the animation
    - press c to capture the animation frame sequence, which is then returned
    as `mov` when the viewer is closed.
    
    NOTE LineSet for wireframe looks junky (cannot make the lines thicker)
    TODO relationship between change_field_of_view and the result??
    
    Args:
        meshes (list): list of `open3d.geometry.Geometry`s
        **kwargs (dict): some visualization options
    Returns:
        mov (ndarray): rgb frames (num_frames, height, width, 3) np.uint8
    """
    defaults = dict(
        bg_color=[255, 255, 255],
        twist_axis='x',
        twist_angle=10,
        num_frames=500,
        pause_factor=4,
        viz_height=640,
        viz_width=640
    )
    bg_color = kwargs.get('bg_color', defaults['bg_color'])
    twist_axis = kwargs.get('twist_axis', defaults['twist_axis'])
    twist_angle = kwargs.get('twist_angle', defaults['twist_angle'])
    num_frames = kwargs.get('num_frames', defaults['num_frames'])
    pause_factor = kwargs.get('pause_factor', defaults['pause_factor'])
    viz_height = kwargs.get('viz_height', defaults['viz_height'])
    viz_width = kwargs.get('viz_width', defaults['viz_width'])

    assert num_frames%4 == 0

    # Set up the visualizer
    class KeyPressManager():
        """Visualizer callbacks for keypress events"""
        def __init__(self):
            self.capture = False
            self.pause = False
        def key_callback_capture(self, vis, action, mods):
            """pressing `c` triggers capturing the next animation sweep"""
            if action == 1:
                print("# capturing the next animation sweep!")
                self.capture = True
        def key_callback_pause(self, vis, action, mods):
            """pressing `space` pauses/resumes the animation"""
            if action == 1:
                self.pause = not self.pause

    viz = open3d.visualization.VisualizerWithKeyCallback()
    kpm = KeyPressManager()
    viz.register_key_action_callback(32, kpm.key_callback_pause)
    viz.register_key_action_callback(67, kpm.key_callback_capture)
    viz.create_window(width=viz_width, height=viz_height, left=50, top=50, visible=True)    
    ro = viz.get_render_option()
    ro.light_on = True
    ro.mesh_show_wireframe = True
    ro.mesh_show_back_face = True
    ro.background_color = np.array(bg_color)/255.

    # Set FOV
    viz.add_geometry(meshes[0])
    viz.get_view_control().change_field_of_view(-5)
    viz.clear_geometries()

    # Set up the mesh and twist animation sequences
    # angular: twist left-right-right-left (returns to the start)
    step_size =twist_angle/(num_frames/2)/(0.003*180/np.pi)  # 0.003 rad/pixel
    seq = [-step_size]*int(num_frames/4) + [step_size]*int(num_frames/2) + [-step_size]*int(num_frames/4)
    # geometries: pause(first geometry)-forward-pause(last geometry)-reverse
    stride = int((num_frames/2)//(len(meshes)+pause_factor))
    pad = int((num_frames/2)%(len(meshes)+pause_factor))
    num_pause = stride*pause_factor+pad
    x = list(itertools.chain(*[[i]*stride for i in range(len(meshes))]))
    meshes_viz = [0]*num_pause+x+[len(meshes)-1]*num_pause+x[::-1]
    meshes_iter = itertools.cycle(meshes_viz)
    this_geom = meshes[next(meshes_iter)]
    viz.add_geometry(this_geom, reset_bounding_box=False)

    print('#----------------------------------------------------------------')
    print('# launching mesh animation with the following properties')
    print('#--------')
    print('# number of meshes        :', len(meshes))
    print('# frames per anim cycle   :', num_frames)
    print('# twist axis              :', twist_axis)
    print('# twist step size    [deg]:', step_size)
    print('# twist angle        [deg]:', twist_angle)
    print('# field of view      [deg]:', viz.get_view_control().get_field_of_view())
    print('#----------------------------------------------------------------')

    # Run it!
    animation_counter = 0
    capture_counter = 0
    mov = None
    while viz.poll_events():
        if kpm.capture and not kpm.pause:
            capture_this_frame = False
            if capture_counter == 0:
                # starting capture
                # crop frames to have even dimensions (cannot resize visualizer)
                pcpi = viz.get_view_control().convert_to_pinhole_camera_parameters().intrinsic                
                mov_height = pcpi.height if pcpi.height%2==0 else pcpi.height-1
                mov_width = pcpi.width if pcpi.width%2==0 else pcpi.width-1
                mov = np.zeros((num_frames, mov_height, mov_width, 3), dtype=np.uint8)
                print('# output mov dims         :', mov.shape)
                print('# output mov raw size [MB]:', np.prod(mov.shape)/1024/1024)
                print('# start capture')
                capture_this_frame = True
            elif 0 < capture_counter < num_frames:
                # continue capture
                capture_this_frame = True
            elif capture_counter == num_frames:
                # we are done
                print()
                print('# done capture')
                print('#--------')
                kpm.capture = False
                capture_counter = 0
            if capture_this_frame:
                print('# capture frame:', capture_counter, end="\r")
                rgb = viz.capture_screen_float_buffer(do_render=False)
                ix = animation_counter%num_frames
                mov[ix] = (np.array(rgb)*255).astype(np.uint8)[:mov_height, :mov_width]
                capture_counter += 1

        if not kpm.pause:
            # Update the mesh
            viz.remove_geometry(this_geom, reset_bounding_box=False)
            this_geom = meshes[next(meshes_iter)]
            viz.add_geometry(this_geom, reset_bounding_box=False)

            if twist_axis == 'y':
                viz.get_view_control().rotate(seq[animation_counter%len(seq)], 0, 0, 0)
            elif twist_axis == 'x':
                viz.get_view_control().rotate(0, seq[animation_counter%len(seq)], 0, 0)
            viz.update_renderer()
            animation_counter += 1
    viz.run()
    viz.destroy_window()
    return mov

def make_decimation_schedule(mx, mn, dec_steps, method='geometric'):
    """decimation schedule"""
    if method == 'geometric':
        out = (mx*(mn/mx)**(np.arange(dec_steps)/(dec_steps-1))).astype(int)
    elif method == 'linear':
        out = np.linspace(mx, mn, dec_steps).astype(int)
    return out

def incremental_decimation(mesh, dec_min, dec_steps, method='geometric'):
    """make a series of decimated meshes without redundant effort"""
    num_tri = np.array(mesh.triangles).shape[0]
    schedule = make_decimation_schedule(num_tri, dec_min, dec_steps, method)
    meshes = [mesh]
    for n in schedule[1:]:
        meshes.append(meshes[-1].simplify_quadric_decimation(n))
    return meshes

def make_trimesh(xyz):
    tri = scipy.spatial.Delaunay(xyz[:, :2])
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(xyz)
    mesh.triangles = open3d.utility.Vector3iVector(tri.simplices)
    mesh = add_bbox_points(mesh)
    return mesh


if __name__ == "__main__":
    pass
