#TODO

1) Dataset normalisation for RL seems to reduce performance, this might just be a function of the learning rate. 
So probably worth tuning for more aggresively.
   
2) Gumbel softmax has been placed in, also doesn't seem to do too great. This might be to do with the fairly limited 
dimensionality it has been given for continuous messages. Probably should do some tuning too. Is the type of noise 
   logical in the discrete setting? Maybe some kind of BSC would be more appropriate. 
   
3) Need to insert independence between the agents. This is likely to make things really bad. But, know it'll work with 
biases. Also need to figure out how to do this for the continuous setting. It might be cleaner to treat the independent
   and continuous setting as two different implementations. Otherwise it is going to get a tad confusing.