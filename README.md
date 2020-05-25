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
In addition to using the widget boxes, points in the diagram can be marked by right clicking.

### Widgets (upper right corner)

### Brightness distribution (lower left corner)

### Screen model (lower right corner)


