clear;
close all;
clc;

%load dataset
data = load('headbrain.txt'); %load dataset from a text file
x = data(:,3); %input training data(head size(cm^3))
y = data(:,4); %labels to data(Brain weight(grams))
l = length(y); %number of training dataset

%data visualization
xa = min(x); 
xb = max(x);
xm = linspace(xa,xb,100);

figure();
plot(x,y,'.b');
hold on;
title('Data Visualization');
xlabel('Head Size(cm^3)');
ylabel('Brain Weight(grams)');
hold off;
axis equal tight;

%splitting into train and test data
x_train = data(1:(2*l/3),3); %input training data(head size(cm^3))
x_test =  data((2*l/3):end,3); %input test data(head size(cm^3))
y_train = data(1:(2*l/3),4); %labels to training data(Brain weight(grams))
y_test = data((2*l/3):end,4); %labels to test data(Brain weight(grams))

l_train = length(y_train);
l_test = length(y_test);

%%Calculate coefficients (slope(m) and intercept(b))
mean_x = mean(x_train);
mean_y = mean(y_train);
num = 0;
den = 0;
for i=1:l_train
    x_diff = (x_train(i,:) - mean_x);
    y_diff = (y_train(i,:) - mean_y);
    num = num + (x_diff * y_diff);
    den = den + (x_diff^2);    
end
m = num/den;
b = mean_y - (m*mean_x);

%%Visualize the predicted output along with actual label for test dataset
xa_train = min(x_train);
xb_train = max(x_train);
xm_train = linspace(xa_train,xb_train,100);

f=b+m*xm;
figure();
plot(x_train,y_train,'.b');
hold on;
plot(xm_train,f,'-r');
title('Linear Regression for Train Data');
xlabel('Head Size(cm^3)');
ylabel('Brain Weight(grams)');
legend('Actual','Predicted');
hold off;
axis equal tight;

%Calculation of R2_Score
sst= 0;
ssr = 0;
for j=1:l
    pred_y = m*x(j,:) + b;
    sst = sst + ((y(j,:)-mean_y)^2);
    ssr = ssr + ((y(j,:)-pred_y)^2);    
end
R2_score = 1-(ssr/sst);
R2_score

%%Calculation of Rooted Mean Squared Error
rmse = sqrt(ssr/l);
rmse

%%Calculate and Compare predicted values

ss = 0;
fprintf('Actual Brain Weight(grams) ----> Predicted Brain Weight(grams)\n');
for k=1:l_test
    pred_y(k) = m*x_test(k,:) + b;   
    ss = ss + ((y_test(k,:)-pred_y(k))^2); 
    err(k) = (sqrt(ss/k));
    fprintf('           %d          ---->         %.2f     \n',y_test(k,:),pred_y(k));   
end
%%Visualize the predicted output along with actual label for test dataset
xa_test = min(x_test);
xb_test = max(x_test);
xm_test = linspace(xa_test,xb_test,100);

f_test=b+m*xm_test;
figure();
plot(x_test,y_test,'.b');
hold on;
plot(xm_test,f_test,'-r');
title('Linear Regression for Test Data');
xlabel('Head Size(cm^3)');
ylabel('Brain Weight(grams)');
legend('Actual','Predicted');
hold off;
axis equal tight;

%%Accuracy and Prediction error curve with varying data size
sst= 0;
ssr = 0;
for j=1:l
    pred_y = m*x(j,:) + b;
    sst = sst + ((y(j,:)-mean_y)^2);
    ssr = ssr + ((y(j,:)-pred_y)^2); 
    accuracy(j) = (1-(ssr/sst))*100;
    error(j) = (sqrt(ssr/j));
    iteration(j) = j;
end
figure();
plot(iteration,accuracy,'-r');
grid on;
hold on;
plot(iteration,error,'-b');
title('Accuracy and Error curve with varying size of dataset');
xlabel('Number of iterations(Data size)');
ylabel('Accuracy(%)');
legend('Accuracy','Error');
hold off;

%%Plotting Error bar
figure();
errorbar(x_test,y_test,err,'.');
hold on;
plot(xm_test,f_test,'-r');
title('Error bar for actual and predicted output');
xlabel('Head Size(cm^3)');
ylabel('Brain Weight(grams)');
legend('Actual','Predicted');
hold off;
axis equal tight;