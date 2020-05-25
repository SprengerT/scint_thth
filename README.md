# scint_thth

## General
This is a graphical interface for the analysis of theta-theta diagrams. Example data and specifications are included.

## Prerequisites
The script is ment to be executed with Python3. Python2 compatibility is not tested. The following non-standard python packages are needed: 
* NumPy       (e.g. `pip3 install --upgrade --user numpy`)
* SciPy       (e.g. `pip3 install --upgrade --user scipy`)
* Matplotlib  (e.g. `pip3 install --upgrade --user matplotlib`)
* ruamel.yaml (e.g. `pip3 install --upgrade --user ruamel.yaml`)

After cloning, there is no installation necessary and a complete set of example data is included, such that the interface could be started immediately. To analyze your own data, you need to adapt __specs.yaml__ accordingly. _(More explanation will be added here)_

## Use
In the download directory, the interface can be started via
```python
python3 scint_thth.py
```
The interface consists of a number of widgets and plots. The idea is to use matplotlib's pan and zoom functions with the left mouse button for navigating and the right mouse button and the widgets to fit a screen model to the theta-theta diagram. In the following, the parts of the interface will be explained.

### Theta-Theta diagram (upper left corner)
This plot shows your data. The red line shows the theoretical shape of the current image interfering with a perfectly one-dimensional screen. Only the horizontal solution is shown, while the corresponding vertical result of the interference with inversed order is hidden. The small red circle shows your currently marked feature. Its location can be changed right here by right clicking on the desired location or by using the widget. White circles show saved features of the same image and white lines show other saved images.

### Brightness distribution (lower left corner)
The red curve shows on the y axis the values of pixels in the theta-theta diagram along the red line there. The values on the x axis are determined by the location on the one-dimensional screen used to compute that line. The black curve shows the median of the brightness distribution of all saved lines as an approximation to the real brightness distribution. This plot can be used to validate an image candidate by checking if the features are at the same place as for the other images.

### Screen model (lower right corner)
This plot shows screen solutions for the current data and parameters, in angular coordinates as viewed from earth. The x axis is aligned with the orientation of the screen, which is indicated by the solid black line. The dashed line indicates the direction of the effective velocity. For each image all solutions are shown. There can be up to two solutions matching the same parameters, from which the first one is marked as a dot and the second one as a cross. The current images is marked in red.

### Widgets (upper right corner)
* __1st row:__ The first row shows the index of the current line (resp. the current image) and buttons to save or delete that line, which means to save or delete the current values of theta and gamma. After saving or deleting, the interface will automatically switch to the next line. If that line does not exist yet, it will create one in the case of saving and switch to the last existing one in the case of deleting.
* __2nd row:__ This row shows the index of the current point (resp. the current feature) and buttons to save and delete it. These widgets work like the corresponding ones for lines.
* __3rd row:__ This row shows the location of the current point in numbers and a button which initializes a fit to find the best image parameters (theta and gamma) for the set of points belonging to the current line. The location of the current point can be either changed here or by right clicking in the theta-theta diagram. When using the fit, care needs to be taken to delete the last automatically created point at (0,0)! The fit only uses the saved points, such that the current coordinates are not regarded by the fit if not saved.
* __4th row:__ This row contains the parameter range of the theta parameter, a button to anchor the reset values to the current values of the sliders and a button to reset the values of all sliders.
* __sliders:__ The sliders can be used to change the three parameters that are relevant to the solutions. The values of theta and gamma determine the location and shape of the line, while beta changes the possible locations of the image on the screen matching this line. The following parameterization is used:

_theta_: absolute value of the coordinate vector of the image

_alpha_: angle of the image relative to the screen axis (0 on the right)

_beta_: angle of the effective velocity relative to the screen axis (0 on the right)

_gamma_: the combination of alpha and beta to which the theta-theta diagram and secondary spectrum are sensitive 

__gamma = cos(beta-alpha)/cos(beta)__

