# avengers-airship-temperature-detection

This project deals with a classification problem of three classes. <br>
In order to solve the problem we implemented an API for all keras pre-trained models.

## Example - Pre-Trained MobileNet with Fine-Tune of a Single FC Layer
We use 4-fold Cross-Validation technique for evaluation, as shown below.

* Classical MobileNet results:
![image](https://user-images.githubusercontent.com/55198967/125076702-e8dfe380-e0c8-11eb-9bc8-eedf428d5b91.png)

* MobileNet variants results:
![image](https://user-images.githubusercontent.com/55198967/125076793-044aee80-e0c9-11eb-9fae-9a15ba03af5e.png)

* For a complete view of all the models we use, the mean of all folds:
![image](https://user-images.githubusercontent.com/55198967/125077071-59870000-e0c9-11eb-9a73-6e55729f3411.png)

* Additionaly, we provide a summary graph of train and validation phases:
![image](https://user-images.githubusercontent.com/55198967/125077868-47f22800-e0ca-11eb-86ed-0f2d7d8fdfc2.png)

* Finnaly, a time-series of the Confusion Matrices will help us understand the success on the test sample:
![image](https://user-images.githubusercontent.com/55198967/125077762-2133f180-e0ca-11eb-805a-fc9b2ae24907.png)

