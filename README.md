# Fluid Simulation

![image](https://github.com/the-other-mariana/fluid-simulation/blob/master/res/reflection.png?raw=true) <br />

## Usage

1. Download this repo and store it in your computer.
2. Open a terminal and go to the root directory of this folder.
3. Make sure you have installed the needed dependencies by typing:

```
$ pip install numpy
$ pip install matplotlib
$ pip install ffmpeg
```

*Note: Go to Install FFmpeg on Windows section if you haven't installed FFmpeg software locally before. It must be added to PATH so that videos can be saved.*

4. Type to run:

```
$ python fluid.py -i config.json
```

Where the config.json file is the input file inside the same folder as main.py file.

## Input

The [config.json](https://github.com/the-other-mariana/fluid-simulation/blob/master/config.json) file is the input file you must provide as a command parameter. The structure of the file must be the following:

1. `color`: string that contains any of the available options in [colors.py](https://github.com/the-other-mariana/fluid-simulation/blob/master/colors.py).

2. `frames`: integer that determines the frame duration of the video.

3. `sources`: an array of dictionaries. Each dictionary in the array represents an emitter, which is a source of density and velocity. There cannot be emitters of just velocity or just density, because it would not make sense. Emitters must contain:

	- `position`: x and y integers, which are the top left position. 
	- `size`: integer that defines an NxN square emitter.
	- `density`: integer that represents the amount of density of the emitter.
	- `velocity`: 
		- x and y float/integer numbers that represent the velocity direction of the emitter. 
		- `behaviour`: string that contains any of the available options in [behaviours.py](https://github.com/the-other-mariana/fluid-simulation/blob/master/behaviours.py).
		- `factor`: float integer/float number that will act as a parameter depending on the behaviour chosen.

4. `objects`: an array of dictionaries. Each dictionary in the array represents an object, where each of the objects must contain:
	- `position`: x and y integers, which are the top left position. 
	- `size`: height and width integers, which will be the shape of a height x width rectangular object.
	- `density`: integer that represents the amount of density of the object. An object is indeed having a constant amount of density that will not be modified by the liquid, since it's a solid, but you need to determine the density or 'color' the object will have visually.

The folder [evidences](https://github.com/the-other-mariana/fluid-simulation/tree/master/evidences) contains a series of example JSON files and their output videos, with both simple and complex examples of the output.

## Features

- **Color Scheme** <br />

Inside the [config.json](https://github.com/the-other-mariana/fluid-simulation/blob/master/config.json) file, change the `color` property and write the color scheme you want from the list below. <br />

![image](https://github.com/the-other-mariana/fluid-simulation/blob/master/res/color-maps-02.png?raw=true)<br />

For example, by having 'hot' as the `color` property in the json file, you get the following: <br />

![image](https://github.com/the-other-mariana/fluid-simulation/blob/master/res/colors-02.gif)<br />

- **Sources Placement** <br />

Inside the [config.json](https://github.com/the-other-mariana/fluid-simulation/blob/master/config.json) file, you can specify the characteristics of an **emitter** you want to place. An emitter is a source of density and certain velocity. <br />

![image](https://github.com/the-other-mariana/fluid-simulation/blob/master/res/cover.gif)<br />

- **Objects Placement** <br />

Inside the [config.json](https://github.com/the-other-mariana/fluid-simulation/blob/master/config.json) file, you can specify the position and shape of a solid object inside the fluid. <br />

![image](https://github.com/the-other-mariana/fluid-simulation/blob/master/res/objects-03.gif)<br />

- **Velocity Behaviours** <br />

Inside the [config.json](https://github.com/the-other-mariana/fluid-simulation/blob/master/config.json) file, change the `behaviour` property inside `velocity` and write the behaviour of the velocity of said emitter you wish for. Supported options are: <br />

1. `zigzag vertical`,

![image](https://github.com/the-other-mariana/fluid-simulation/blob/master/res/objects-03.gif)

2. `zigzag horizontal`, that works the same as the above but horizontally.

3. `vortex`, 

![image](https://github.com/the-other-mariana/fluid-simulation/blob/master/res/vortex-example.gif)

4. `noise`,

![image](https://github.com/the-other-mariana/fluid-simulation/blob/master/res/noise-example.gif)

5. `fourier` (left), which is a bit like a zigzag (right) but noisier.

![image](https://github.com/the-other-mariana/fluid-simulation/blob/master/res/fourier-example.gif)

6. `motor`

![image](https://github.com/the-other-mariana/fluid-simulation/blob/master/res/motor-example.gif)

## Install FFmpeg on Windows

 Apart from the pip installation of ffmpeg, you need to install ffmpeg for your machine OS (in my case, Windows 10) by going to either of the following links:

- [ffmpeg.org](https://ffmpeg.org/download.html)
	- Click on the Windows icon.
	- Click on gyan dev option.
- [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)
	- Go to the Git section and click on the first link.
	- Extract the folder from the zip.
	- Cut and paste the folder in your C: disk.
	- Add C:\FFmpeg\bin to PATH by typing in a terminal with admin rights: <br />

	```
	$ setx /m PATH "C:\FFmpeg\bin;%PATH%"
	```
	- Open another terminal and test the installation by typing: <br />

	```
	$ ffmpeg -version
	```
## Handy Links

- [Normalize Color in Matplotlib](https://stackoverflow.com/questions/48228692/maintaining-one-colorbar-for-maptlotlib-funcanimation) <br />

- [Fourier Series Square Wave](https://mathworld.wolfram.com/FourierSeriesSquareWave.html) <br />