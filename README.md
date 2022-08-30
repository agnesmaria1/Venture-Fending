# Venture-Fending

We are building neural network with multiple neuron layers between the model’s input and output layers. The deep neural network we build will help a venture fund predict which startups will become financially successful.

---

## Technologies

This project leverages python 3.9 with the following package:

* [Pandas](https://pandas.pydata.org/) - Entry point, open source data analysis and manipulation tool.

* [Keras](https://keras.io/about/) - Deep learning API written in Python, running on top of the machine learning platform TensorFlow.

* [TensorFlow](https://www.tensorflow.org/) - An end-to-end, open-source machine learning platform.

---

## Installation Guide

Before running the application first install the following dependencies on your GitBash, macOS user, please upload the notebook using Google Colab.

```python
pip install --upgrade tensorflow
python -c "import tensorflow as tf;print(tf.__version__)"
python -c "import tensorflow as tf;print(tf.keras.__version__)"
```

Next, import required libraries and dependencies.

```python
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
```

---

## Usage

To use this application simply clone the repository and run the **venture_funding_with_deep_learning.ipynb** with:

```python
  venture_funding_with_deep_learning.ipynb
```
Or if you are MacOS user, please open [Google Colab](https://colab.research.google.com/drive/1g2ffc25y9OZ8ik3htTlF6k36fzyKTMoc?usp=sharing)

---

## Contributors

This project brought to you by Agnes.

---

## License
[MIT](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt)
