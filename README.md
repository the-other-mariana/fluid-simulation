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

*Note: Go to Install FFmpeg on Windows section if you haven't installed FFmpeg software locally before.*

4. Type to run:

```
$ python fluid.py -i config.json
```

## Features

- **Color Scheme** <br />

Inside the [config.json](https://github.com/the-other-mariana/fluid-simulation/blob/master/config.json) file, change the `color` property and write the color scheme you want from the list below. <br />

![image](https://github.com/the-other-mariana/fluid-simulation/blob/master/res/color-maps-02.png?raw=true)<br />

For example, by having 'hot' as the `color` property in the json file, you get the following: <br />

![image](https://github.com/the-other-mariana/fluid-simulation/blob/master/res/colors-02.gif)<br />

- **Sources Placement** <br />

Inside the [config.json](https://github.com/the-other-mariana/fluid-simulation/blob/master/config.json) file, you can specify the characteristics of an **emitter** you want to place. An emitter is a source of density and certain velocity. <br />

![image](https://github.com/the-other-mariana/fluid-simulation/blob/master/res/cover.gif)<br />

- **Object Placement** <br />

Inside the [config.json](https://github.com/the-other-mariana/fluid-simulation/blob/master/config.json) file, you can specify the position and shape of a solid object inside the fluid. <br />

![image](https://github.com/the-other-mariana/fluid-simulation/blob/master/res/objects-03.gif)<br />

- **Velocity Behaviours** <br />

Inside the [config.json](https://github.com/the-other-mariana/fluid-simulation/blob/master/config.json) file, change the `behaviour` property inside `velocity` and write the behaviour of the velocity of said emitter you wish for. Supported options are: <br />

- `zigzag vertical`,

![image](https://github.com/the-other-mariana/fluid-simulation/blob/master/res/objects-03.gif)

- `zigzag horizontal`, that works the same as the above but horizontally.

- `vortex`, 

![image](https://github.com/the-other-mariana/fluid-simulation/blob/master/res/vortex-example.gif)

- `noise`,

![image](https://github.com/the-other-mariana/fluid-simulation/blob/master/res/noise-example.gif)

- `fourier` (left), which is a bit like a zigzag (right) but noisier.

![image](https://github.com/the-other-mariana/fluid-simulation/blob/master/res/fourier-example.gif)

- `motor`

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