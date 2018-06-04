by far, the best results are

```
root:INFO:2018-06-03 17:26:34,344:Loss sum : 414.216461 
root:INFO:2018-06-03 17:26:35,105:Total validation count 1908
root:INFO:2018-06-03 17:26:35,164:Precision @ 100: 0.780000
root:INFO:2018-06-03 17:26:35,221:Precision @ 200: 0.675000
root:INFO:2018-06-03 17:26:35,279:Precision @ 300: 0.616667
root:INFO:2018-06-03 17:26:36,900:ROC-AUC score: 0.973260
root:INFO:2018-06-03 17:26:38,281:Average Precision: 0.177625
```

Parameters are:
- Hidden dim = 256
- position dim = 5
- learning rate = 1e-4
- weight-decay = 1e-5
- num layers = 2
- num epoch = 3
