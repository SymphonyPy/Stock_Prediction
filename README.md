# Stock_Prediction
An LSTM neural network model writen by Tensorflow.

------------
**RUN**

Just run train.py with`python train.py`
And then use `tensorboard --logdir=(tensorboard data file)` 
like `tensorboard --logdir=2018-04-21` to launch Tensorboard in your browser.

Also, you can use automaticly saved model to run a demonstration of prediction. Comment out (I googled this word) the training cod in `if __name__ == "__main__":`, with only `demonstrate()` left. It is surprised that the result seems good.

------------

**PROBLEMS**
1. it is weired that accuracy of test set is much higher than that of train set, 70%+ and 40%+ respectly. I have looked it upon google and zhihu, it is said that the reason is that the amount of train set is too big and it will take a while for training to make it act as normal. And right now I am working on it.
2. The macro-F1 is really low.

------------

**THANKS**

[@PTSTS](https://github.com/PTSTS "@PTSTS") for dataset and advice for model.
