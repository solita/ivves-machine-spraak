# Machine Spraak - Methods of Audio Analysis in Python

## Introduction

This course gives a short introduction to audio analysis from scratch with Python. Our aim is to give a better understanding of what are the fundamental building blocks of
audio analysis (and signal processing in general), how choices made at each of these steps affects the result and also to shed light on what some more complex features actually
tell us about the original signal. As such, we avoid using libraries with ready-made solutions (such as [librosa](https://librosa.org)) and instead build everything from scratch
with mainly `numpy` and `SciPy`.

By the end of this 3-part course you will have learnt how an analogue signal is transformed into a digital format so that we can begin analysing it, how rapidly oscillating signals
become more understandable after we separate their frequency components and how we can then extract many rich features out of this frequency representation. We'll also look at two
classical methods of analytically recovering structure from temporally ordered signals, either in the frequency or time directions, in order to solve some interesting problems.

There is some mathematical notation sprinkled around these notebooks, but for the most part these are not essential for understanding the material so feel free to skip them.
The main focus is on actively experimenting with the code of the notebooks to see how different changes affect the output. Sometimes this is aided via `IPython` widgets or
specific exercises (indicated with the 🔊 icon), but you are also encouraged to go beyond these suggestions.

Have fun and I hope you will learn something new and useful!

## Lessons

This course consists of three lessons in the form of Jupyter notebooks that you
can run in Google Colab via the links below (or clone the repo and run
locally).
In order to make the widgets work you should always choose `Runtime -> Run all`
from the Colab menu before proceeding with each notebook.

- [Lesson 1 - Digital Signal Processing and Spectral Analysis](https://colab.research.google.com/github/solita/ivves-machine-spraak/blob/main/courses/methods/Methods-1.ipynb)

- [Lesson 2 - Cepstral Analysis, Harmonics and the Mel-scale](https://colab.research.google.com/github/solita/ivves-machine-spraak/blob/main/courses/methods/Methods-2.ipynb)

- [Lesson 3 - Self-similarity, Kernels and Structural Analysis](https://colab.research.google.com/github/solita/ivves-machine-spraak/blob/main/courses/methods/Methods-3.ipynb)
