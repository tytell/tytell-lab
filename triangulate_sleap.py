import os
import aniposelib
import argparse
import yaml
import cv2

import numpy as np
import matplotlib

import matplotlib.pyplot as plt

from sleap.io.dataset import Labels
import pandas as pd
import re
import fnmatch

from copy import copy, deepcopy
from warnings import warn

from contextlib import contextmanager
@contextmanager
def VideoCapture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()

def calibrate_charuco(board, videonames, camnames, outfile):
    camgroup = aniposelib.cameras.CameraGroup.from_names(camnames)

    err, points = camgroup.calibrate_videos(videonames, board)
    camgroup.dump(outfile)

    print(points)

    if (err > 4):
        warn(f'Large error ({err})!')
    
    # collate the points into a dataframe
    boardpts = []
    for pts1, cam1 in zip(points, camnames):
        col_idx = pd.MultiIndex.from_tuples([[cam1, 'x'], [cam1, 'y']],
                                            names=['camera', 'var'])

        idx = np.empty((0,2))
        c = []
        for frdata in pts1:
            if frdata['corners'].shape[0] == 0:
                next

            c.append(frdata['corners'].reshape((-1, 2)))

            idx1 = np.empty((frdata['ids'].shape[0], 2))
            idx1[:,0] = frdata['framenum'][1]
            idx1[:,1] = frdata['ids'][:,0]

            idx = np.row_stack((idx, idx1))

        idx = pd.MultiIndex.from_arrays([idx[:,0], idx[:,1]], 
                                        names=['frame', 'point'])

        boardpts.append(pd.DataFrame(np.concatenate(c), index=idx, columns=col_idx))

    # only keep the matching points (same frame and id)
    boardpts = pd.concat(boardpts, axis=1, join='inner')

    return camgroup, boardpts

def refine_calibration(camgroup, pts, max_err, nodes='all', outfile=None):
    npt = pts.shape[0]
    good = ~pts.loc[:,(slice(None), ['x', 'y'])].isna().any(axis=1)
    pts = pts.loc[good, :]
    print(f'{pts.shape[0]}/{npt} points visible in all cameras')

    if nodes != 'all':
        pts = pts.loc[(slice(None), slice(None), nodes), :]
        print('{}/{} ({:2.0f}%) points in selected nodes'.format(pts.shape[0], npt, np.round(pts.shape[0]/npt*100)))

    good = pts.loc[:, (slice(None), 'err')].max(axis=1, skipna=False) < max_err

    pts = pts.loc[good, (camgroup.get_names(), ['x', 'y'])]
    print('{}/{} ({:2.0f}%) points below maximum error'.format(pts.shape[0], npt, np.round(pts.shape[0]/npt*100)))

    calibpts = pts.to_numpy(dtype=float)
    calibpts = np.reshape(calibpts, [-1, 3, 2]).transpose((1, 0, 2))

    camgroup1 = deepcopy(camgroup)
    camgroup1.bundle_adjust_iter(calibpts, extra=None,
                            n_iters=6, start_mu=15, end_mu=1,
                            max_nfev=200, ftol=1e-4,
                            n_samp_iter=200, n_samp_full=1000,
                            error_threshold=0.3, only_extrinsics=False,
                            verbose=True)    

    if outfile is not None:
        camgroup1.dump(outfile)

    return camgroup1

def triangulate_points(camgroup, pts):
    # get rid of any columns that might have come from a previous reprojection
    pts = pts.loc[:, (camgroup.get_names(), ['x','y'])]

    # now compute the reprojection error
    ptsmatrix = pts.to_numpy().reshape((-1, 3, 2))
    ptsmatrix = ptsmatrix.transpose((1,0,2)).astype(np.float64)

    pts3d = camgroup.triangulate(ptsmatrix, progress=True, undistort=True)

    col_idx = pd.MultiIndex.from_product([['3D'], ['x', 'y', 'z']],
                                         names=['camera','var'])
    dfpts3d = pd.DataFrame(pts3d, index=pts.index, 
                              columns=col_idx)
    pts = pd.concat((pts, dfpts3d), axis=1)

    reproj = camgroup.project(pts3d)
    reproj = reproj.transpose((1,0,2))
    reproj = reproj.reshape((-1, 6))

    col_idx = pd.MultiIndex.from_product([camgroup.get_names(), ['Xr', 'Yr']],
                                         names=['camera','var'])

    reproj = pd.DataFrame(reproj, 
                          index=pts.index, 
                          columns=col_idx)

    pts = pd.concat((pts, reproj), axis=1)

    errs = []
    for cam in camgroup.get_names():
        xy = pts.loc[:,(cam,['x','y'])].to_numpy(dtype=float)
        XYr = pts.loc[:,(cam,['Xr','Yr'])].to_numpy(dtype=float)

        errs.append(
            pd.DataFrame(np.linalg.norm(XYr - xy, axis=1), index=pts.index,
                    columns=pd.MultiIndex.from_product([[cam], ['err']],
                                                names=['camera','var']))
        )

    errs = pd.concat(errs, axis=1)
    pts = pd.concat((pts, errs), axis=1)

    return pts

def get_basis(axesdata, order_axes='length', make_right_handed=True):
    origin = axesdata.iloc[0]

    axesdata0 = axesdata.iloc[1:] - origin.values

    axesdata0.index = pd.Index(['x', 'y', 'z'])
    axesdata0.columns = pd.Index(['xorig', 'yorig', 'zorig'])

    if order_axes == 'length':
        axlen = np.sqrt(np.square(axesdata0.T).sum())
        axlen = axlen.sort_values(ascending=False)

        axesdata0 = axesdata0.reindex(axlen.index)

        axlenstr = ['{}={:.1f}'.format(n,v) for n,v in zip(axlen.index, axlen)]
        print('Ordering axes by length: {}'.format(', '.join(axlenstr)))
    elif order_axes is None:
        pass

    h = np.cross(axesdata0.iloc[0], axesdata0.iloc[1]) @ axesdata0.iloc[2]
    if make_right_handed and (h < 0):
        print('Axes matrix is left handed! Flipping {} axis'.format(axesdata0.index[2]))

        axesdata0.iloc[2] = -axesdata0.iloc[2]

    R = gram_schmidt(axesdata0.T.to_numpy())

    R = pd.DataFrame(R, columns = axesdata0.index)

    # this reorders the columns in alphabetical order
    R = R.reindex(columns=R.columns.sort_values())

    return origin, R

def gram_schmidt(A):
    """Orthogonalize a set of vectors stored as the columns of matrix A."""
    # Get the number of vectors.

    assert(A.ndim == 2)

    n = A.shape[1]
    for j in range(n):
        # To orthogonalize the vector in column j with respect to the
        # previous vectors, subtract from it its projection onto
        # each of the previous vectors.
        for k in range(j):
            A[:, j] -= np.dot(A[:, k], A[:, j]) * A[:, k]
        A[:, j] = A[:, j] / np.linalg.norm(A[:, j])
    return A

def rename_columns(df, columns, inplace=False):
    """Rename dataframe columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe.
    columns : dict-like
        Alternative to specifying axis. If `df.columns` is
        :obj: `pandas.MultiIndex`-object and has a few levels, pass equal-size tuples.

    Returns
    -------
    pandas.DataFrame or None
        Returns dataframe with modifed columns or ``None`` (depends on `inplace` parameter value).
    
    Examples
    --------
    >>> columns = pd.Index([1, 2, 3])
    >>> df = pd.DataFrame([[1, 2, 3], [10, 20, 30]], columns=columns)
    ...     1   2   3
    ... 0   1   2   3
    ... 1  10  20  30
    >>> rename_columns(df, columns={1 : 10})
    ...    10   2   3
    ... 0   1   2   3
    ... 1  10  20  30
    
    MultiIndex
    
    >>> columns = pd.MultiIndex.from_tuples([("A0", "B0", "C0"), ("A1", "B1", "C1"), ("A2", "B2", "")])
    >>> df = pd.DataFrame([[1, 2, 3], [10, 20, 30]], columns=columns)
    >>> df
    ...    A0  A1  A2
    ...    B0  B1  B2
    ...    C0  C1
    ... 0   1   2   3
    ... 1  10  20  30
    >>> rename_columns(df, columns={("A2", "B2", "") : ("A3", "B3", "")})
    ...    A0  A1  A3
    ...    B0  B1  B3
    ...    C0  C1
    ... 0   1   2   3
    ... 1  10  20  30
    """
    columns_new = []
    for col in df.columns.values:
        if col in columns:
            columns_new.append(columns[col])
        else:
            columns_new.append(col)
    columns_new = pd.Index(columns_new, tupleize_cols=True)

    if inplace:
        df.columns = columns_new
    else:
        df_new = df.copy()
        df_new.columns = columns_new
        return df_new

def rotate_axes(data, origin, R):
    data3d = data.loc[:, ('3D', ['x', 'y', 'z'])]

    data_rot = rename_columns(data, {('3D', 'x'): ('3D', 'x0'),
                                       ('3D', 'y'): ('3D', 'y0'),
                                       ('3D', 'z'): ('3D', 'z0')})
    
    data3d_r = (data3d.to_numpy() - origin.values) @ R

    col_idx = pd.MultiIndex.from_product([['3D'], ['x','y','z']])
    data3d_r = pd.DataFrame(data3d_r.values, index=data_rot.index, columns=col_idx)

    return pd.concat((data_rot, data3d_r), axis=1)

def plot_reprojected_points(pts, i, videonames=None, videopath=None, zoom=True):
    fr = pts.iloc[i:i+1].index.get_level_values('frame')[0]
    try:
        vid = pts.iloc[i:i+1].index.get_level_values('video')[0]
        q = f'frame == {fr} and video == "{vid}"'
    except KeyError:
        q = f'frame == {fr}'

    print(q)
    pts1 = pts.query(q)

    camnames = list(pts1.columns.get_level_values('camera').unique())
    camnames = [cn1 for cn1 in camnames if cn1 != '3D']

    if videonames is None:
        videoname = list(pts.iloc[i:i+1].index.get_level_values('video'))[0]
        videonames = []
        for cam1 in camnames:
            fn1, _ = re.subn('CAMERA', cam1, videoname)
            videonames.append(fn1)
    
    if videopath is not None:
        videonames = [[os.path.join(videopath, vn1)] for vn1 in videonames]

    try:
        cap = [cv2.VideoCapture(vid1[0]) for vid1 in videonames]

        fig, ax = plt.subplots(ncols=len(cap), nrows=1) #, sharex=True, sharey=True)

        for cam1, cap1, ax1 in zip(camnames, cap, ax):
            cap1.set(1, fr)
            ret, frame1 = cap1.read()

            ax1.imshow(frame1)

            x1 = pts1[(cam1, 'x')].array
            y1 = pts1[(cam1, 'y')].array
            xr1 = pts1[(cam1, 'Xr')].array
            yr1 = pts1[(cam1, 'Yr')].array

            ax1.plot(x1,y1, 'ro')
            ax1.plot(xr1,yr1, 'y+')
            ax1.plot(np.vstack((x1, xr1)), 
                     np.vstack((y1, yr1)), 'y-')

            xx = pts1.loc[(slice(None)), (cam1, ['x','Xr'])].stack()
            yy = pts1.loc[(slice(None)), (cam1, ['y','Yr'])].stack()

            if zoom:
                ax1.set_xlim(pd.concat([xx.min(), xx.max()]).to_numpy() + np.array([-50, 50]))
                ax1.set_ylim(pd.concat([yy.min(), yy.max()]).to_numpy() + np.array([-50, 50]))        
            ax1.invert_yaxis()
            ax1.axis('off')
    finally:
        for c1 in cap:
            c1.release()
    
    return fig

def separate_video_and_camera(vidname, camnames):
    fn1 = re.sub(r'\\', '/', vidname)
    fn1 = os.path.basename(fn1)

    for cam1 in camnames:
        fn1, nsub = re.subn(cam1, 'CAMERA', fn1)
        if nsub == 1:
            matched_camera = cam1
            break
    else:
        matched_camera = None

    return fn1, matched_camera

def load_sleap_points(sleapfiles, calibfile, match_video=None):
    camgroup = aniposelib.cameras.CameraGroup.load(calibfile)
    camnames = camgroup.get_names()

    labels = [Labels.load_file(fn) for fn in sleapfiles]

    node_count = len(labels[0].skeletons[0].nodes)
    node_names = [node.name for node in labels[0].skeletons[0].nodes]

    ptsall = []
    for l1, cam1 in zip(labels, camnames):
        pts = []
        videos_done = []
        for v1 in l1.videos:
            v1name = v1.backend.filename
            if v1name in videos_done:
                warn(f'Video {v1name} in Sleap file multiple times')
                continue
            
            if match_video is not None and not fnmatch.fnmatch(v1name, match_video):
                continue
            
            videos_done.append(v1name)

            vidname1, camname1 = separate_video_and_camera(v1name, camnames)
            if camname1 != cam1:
                continue
            
            frames = l1.get(v1)
            frame_idx = [lf.frame_idx for lf in frames]

            col_ind = pd.MultiIndex.from_product([[camname1], ['x', 'y']],
                                                names = ['camera', 'var'])
            row_ind = pd.MultiIndex.from_product([[vidname1], frame_idx, node_names], 
                                                names = ['video', 'frame', 'node'])

            pts1 = pd.DataFrame(index = row_ind, columns=col_ind)

            for lf in frames:
                if len(lf.user_instances) == 1:
                    inst = lf.user_instances[0]
                elif len(lf.predicted_instances) == 1:
                    inst = lf.predicted_instances[0]
                else:
                    print("Error!")
                    assert(False)
                
                pts1.loc[(vidname1, lf.frame_idx, slice(None)), (camname1, slice(None))] = inst.numpy()

            pts.append(pts1)
        
        pts = pd.concat(pts, axis=0)
        if pts.index.has_duplicates:
            warn('Dropping duplicate videos')

            # this annoyingly complicated expression keeps all unduplicated elements and the first duplicated one
            not_duplicated = ~(ptsall[2].index.duplicated(keep=False) & ptsall[2].index.duplicated(keep='last'))
            pts = pts.loc[not_duplicated]

        ptsall.append(pts)

    ptsall = pd.concat(ptsall, axis=1, join='inner')

    print('Found {} frames across {} matched videos'.format(len(ptsall.index.get_level_values('frame')), 
                                                            len(ptsall.index.get_level_values('video').unique())))    


    return camgroup, ptsall

# Function to separate the name of a video and the camera names, so that we can match up videos for different cameras
def get_trial_and_camname(vidname, camnames, vid_tab):
    if vid_tab is None:
        fn1 = re.sub(r'\\', '/', vidname)
        fn1 = os.path.basename(fn1)

        for cam1 in camnames:
            fn1, nsub = re.subn(cam1, 'CAMERA', fn1)
            if nsub == 1:
                matched_camera = cam1
                break
        else:
            matched_camera = None

        return fn1, matched_camera
    else:
        trials = vid_tab.iloc[:,0]
        names = vid_tab.drop(columns=vid_tab.columns[0])

        tr1 = vidname

        matched_camera = None
        for cam1, allcamnames1 in names.items():
            for idx, n1 in allcamnames1.items():
                if n1 in vidname:
                    tr1 = trials[idx]
                    matched_camera = cam1
                    break

            if matched_camera is not None:
                break
        
        return tr1, matched_camera



def main():
    matplotlib.use('Agg')

    parser = argparse.ArgumentParser(description='Calibrate based on images of Charuco boards')

    parser.add_argument('config', nargs="+")

    parser.add_argument('--base_path', default='.',
                        help='Base path for data files')

    parser.add_argument('-nx', '--nsquaresx', type=int, 
                        help='Number of grid squares horizontally',
                        default=6)
    parser.add_argument('-ny', '--nsquaresy', type=int, 
                        help='Number of grid squares vertically',
                        default=6)
    parser.add_argument('-sz', '--square_length', type=float,
                        help = 'Size of square in mm',
                        default = 24.33)
    parser.add_argument('-mlen', '--marker_length', type=float,
                        help='Size of the Aruco marker in mm',
                        default=17)
    parser.add_argument('-mbits', '--marker_bits', type=int,
                        help='Information bits in the markers',
                        default=5)
    parser.add_argument('-dict','--dict_size', type=int,
                        help='Number of markers in the dictionary',
                        default=50)

    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help='Display increasingly verbose output')

    parser.add_argument('--calibration_videos', action='extend', nargs='+',
                        help="Video files containing synchronized images of the Charuco board")

    parser.add_argument('--camera_names', action='extend', nargs='+',
                        help="Names for eaech of the cameras, in the same order as the videos")

    parser.add_argument('--force_calibration', default=False, action="store_true",
                        help="Run the calibration even if the calibration TOML file is present")

    parser.add_argument('--debug', type=bool, default=False,
                        help="Save debug images for the calibration")
    parser.add_argument('--ndebugimages', type=int, default=10,
                        help="Save debug images for the calibration")

    parser.add_argument('--showreprojection', type=bool, default=False,
                        help="Show debug images to test reprojection error")

    parser.add_argument('--calibration_file', help='TOML File to store the calibration')

    parser.add_argument('-s', '--sleap_files', action='extend', nargs='+',
                        help="SLP files containing points detected using Sleap.ai")
    parser.add_argument('--video_table', default=None,
                        help="CSV file containing a table of matching video files")
    
    parser.add_argument('-o', '--output_file', 
                        help="Name of the output CSV file with the triangulated points")

    args = parser.parse_args()

    if args.config is not None:
        for config1 in args.config:
            with open(config1, 'r') as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)
        
        args = parser.parse_args()

    ## Run the calibration if necessary
    calib_file = os.path.join(args.base_path, args.calibration_file)

    if args.force_calibration or not os.path.exists(calib_file):
        if args.verbose > 0:
            if not os.path.exists(calib_file):
                print(f"No calibration file found. Running calibration")
            else:
                print("force_calibration is True. Running calibration")

        board = aniposelib.boards.CharucoBoard(squaresX=args.nsquaresx,
                                            squaresY=args.nsquaresy,
                                            square_length=args.square_length,
                                            marker_length=args.marker_length,
                                            marker_bits=args.marker_bits,
                                            dict_size=args.dict_size)

        assert len(args.calibration_videos) == len(args.camera_names), \
            f'Number of calibration videos {len(args.calibration_videos)} is different than number of camera names {len(args.camera_names)}'
        camgroup = aniposelib.cameras.CameraGroup.from_names(args.camera_names)

        vidnames = [[os.path.join(args.base_path, vid)] for vid in args.calibration_videos]
        err, rows = camgroup.calibrate_videos(vidnames, board, 
                                init_intrinsics=True, init_extrinsics=True, 
                                verbose=args.verbose > 0)
        
        if args.verbose == 2:
            # save the detected points
            import json
            from copy import deepcopy

            rows_out = deepcopy(rows)

            for rows_cam in rows_out:
                for rows1 in rows_cam:
                    for k, v in rows1.items():
                        if isinstance(v, np.ndarray):
                            rows1[k] = rows1[k].tolist()

            with open('rows-output.json', 'w') as f:
                json.dump(rows_out, f)
                    
        camgroup.dump(calib_file)

        if args.debug:
            # generate images showing the detected points
            for vid, rows1 in zip(args.calibration_videos, rows):
                vid = os.path.join(args.base_path, vid)

                pn, fn = os.path.split(vid)
                fn, _ = os.path.splitext(fn)

                ngoodframes = len(rows1)

                with VideoCapture(vid) as cap:
                    for i in np.linspace(0, ngoodframes, num=args.ndebugimages, endpoint=False).astype(int):
                        fr = rows1[i]['framenum'][1]

                        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                        ret, frame = cap.read()

                        fig, ax = plt.subplots()
                        ax.imshow(frame)
                        ax.plot(rows1[i]['corners'][:,0,0], rows1[i]['corners'][:,0,1], 'ro')

                        outname1 = os.path.join(pn, '{0}-debug-{1:03d}.png'.format(fn, fr))
                        plt.savefig(outname1)
                        plt.close(fig)
    else:
        if args.verbose > 0:
            print(f"Loading calibration from {calib_file}")

        camgroup = aniposelib.cameras.CameraGroup.load(calib_file)

        if args.verbose > 0:
            print(f"Found {len(camgroup.cameras)} cameras")

        assert len(camgroup.cameras) == len(args.camera_names), \
            f'Number of cameras in calibration file {calib_file} ({len(camgroup.cameras)}) is different than number of camera names ({len(args.camera_names)})'

    ## Load in the Sleap data and triangulate
        
    if args.verbose > 0:
        print(f"Loading {len(args.sleap_files)} sleap data files")

    labels = [Labels.load_file(os.path.join(args.base_path, fn)) for fn in args.sleap_files]

    node_count = len(labels[0].skeletons[0].nodes)
    node_names = [node.name for node in labels[0].skeletons[0].nodes]

    videos = [[v1.backend.filename for v1 in l1.videos if re.search(c1, v1.backend.filename) is not None] \
            for l1, c1 in zip(labels, args.camera_names)]

    if args.verbose >= 2:
        print("All videos:\n")
        print(videos)

    ## Match videos from the same trial and different cameras
    vid_tab = None
    if args.video_table is not None:
        vid_tab_file = os.path.join(args.base_path, args.video_table)

        if not os.path.exists(vid_tab_file):
            if args.verbose > 0:
                print(f"Video table file {vid_tab_file} not found. Attempting to match videos by name")
        else:
            vid_tab = pd.read_csv(vid_tab_file)

    # pull out the x and y coordinates from the Sleap data files and match the same point in the same frame across cameras
    if args.verbose > 0:
        print("Extracting points...")
    
    ptsall = []
    already_processed = []
    for l1, cam1 in zip(labels, args.camera_names):
        pts = []
        for v1 in l1.videos:
            trial1, camname1 = get_trial_and_camname(v1.backend.filename, args.camera_names, vid_tab)
            if camname1 != cam1:
                continue
            
            if v1.backend.filename in already_processed:
                print(f"File {v1.backend.filename} seems to be repeated. Skipping...")
                continue
            else:
                already_processed.append(v1.backend.filename)

            if args.verbose > 0:
                print(f"Camera {camname1}, Trial {trial1}")

            frames = l1.get(v1)
            frame_idx = [lf.frame_idx for lf in frames]

            col_ind = pd.MultiIndex.from_product([[camname1], ['x', 'y']],
                                                names = ['camera', 'point'])
            row_ind = pd.MultiIndex.from_product([[trial1], frame_idx, node_names], 
                                                names = ['video', 'frame', 'node'])

            pts1 = pd.DataFrame(index = row_ind, columns=col_ind)

            for lf in frames:
                if len(lf.user_instances) == 1:
                    inst = lf.user_instances[0]
                elif len(lf.predicted_instances) == 1:
                    inst = lf.predicted_instances[0]
                else:
                    assert False, "No instances or multiple instances in frame"
                
                pts1.loc[(trial1, lf.frame_idx, slice(None)), (camname1, slice(None))] = inst.numpy()

            pts.append(pts1)
        
        pts = pd.concat(pts, axis=0)
        duplicates = pts.index.duplicated(keep=False)
        if any(duplicates):
            keep = ~pts.index.duplicated(keep='first')
            if args.verbose > 0:
                print(f"Camera {cam1}: {sum(duplicates)} duplicated frames. Keeping {sum(keep)} ({sum(keep)/len(keep)*100:.0f}%)")
            pts = pts[keep]

        ptsall.append(pts)

    ptsall = pd.concat(ptsall, axis=1)

    if args.verbose > 0:
        print("First rows in the extracted points:")
        print(ptsall.head())

    ptsmatrix = ptsall.to_numpy().reshape((-1, 3, 2))
    ptsmatrix = ptsmatrix.transpose((1,0,2)).astype(np.float64)

    # Triangulate the points to 3D
    if args.verbose > 0:
        print(f"Triangulating {ptsmatrix.shape[1]} points...")

    pts3d = camgroup.triangulate(ptsmatrix, progress=True, undistort=True)

    # and compute the reprojection error
    if args.verbose > 0:
        print("Computing reprojection error...")

    reproj_err = camgroup.reprojection_error(pts3d, ptsmatrix, mean=False)
    errors_norm = np.linalg.norm(reproj_err, axis=2)

    # now build up the same data frame as the original points so that we can merge them all
    # first for the points
    col_ind = pd.MultiIndex.from_product([['3D'], ['x', 'y', 'z']],
                                            names = ['camera', 'point'])

    pts3d = pd.DataFrame(pts3d, index=ptsall.index, columns=col_ind)

    # and for the reprojection error
    col_ind = pd.MultiIndex.from_product([args.camera_names, ['reproj_err']],
                                            names = ['camera', 'point'])

    errors_norm = pd.DataFrame(errors_norm.T, index=ptsall.index, columns=col_ind)

    # print the reprojection errors

    if args.verbose > 0:
        print("Median reprojection errors:")
        print(errors_norm.groupby(level=2).median())

    ptsall3d = pd.concat((ptsall, errors_norm, pts3d), axis=1)

    # rearrange the multiindex so that we can save in a simple CSV format
    ptsall3d.columns = ['_'.join(reversed(c)) for c in ptsall3d.columns.to_flat_index()]
    ptsall3d.reset_index(inplace=True)

    # and save
    if args.verbose > 0:
        print(f"Saving points to {args.output_file}")

    ptsall3d.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    main()

