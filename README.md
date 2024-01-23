# tytell-lab
Shared code for the Tytell lab at Tufts University

## charuco-to-easywand

Contains scripts for detecting points on a [Charuco board](https://docs.opencv.org/3.4/da/d13/tutorial_aruco_calibration.html) using
[Aniposelib](https://anipose.readthedocs.io/en/latest/aniposelib-tutorial.html), then converting them into a data set suitable to
import and calibrate using [EasyWand5](https://biomech.web.unc.edu/wand-calibration-tools/). The calibration can then be used with
[DLTdv8](https://biomech.web.unc.edu/dltdv/) for semi-automatic tracking.

## triangulate_sleap

Calibrates multi-camera views using a Charuco board, then triangulates points from each view to 3D, based on points tracked through [Sleap](https://sleap.ai/index.html)

## Other code

* R/read_labchart.R: Script to process text files exported from ADI LabChart
