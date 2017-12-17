Did is as part of the challenge to Siraj's Youtube video https://www.youtube.com/watch?v=ftMq5ps503w

It predicts a google stock price from the taking into consideration past days data. I have used LSTM, on Tensorflow with Keras on top.

I will hopefully improve this by adding sentiment knowledge to it, by using headlines of the stock for each day.
It should not be much of an extension, since after getting the sentiment score from the headline about the stock, we can basically just add it as a new column to our matrix and our model learns it as a new feature.

Although I haven't fully optimezed the hyper-parameters for now.
