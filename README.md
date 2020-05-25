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

### Widgets (upper right corner)


