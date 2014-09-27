function model = nb_train(Xtrain, Ytrain)
row =  size(Xtrain,1);
column =size(Xtrain, 2);
model=zeros(column,2);
%count the probability of Y=0/Y=1 
count=0;
for i = 1:row
    if Ytrain(i,1) == 0
       count = count + 1;
    end
end
P0 = log10(count/row);
P1 = log10((row-count)/row);

Xtrain0 = Xtrain(Ytrain == 0,:);
Xtrain1 = Xtrain(Ytrain == 1,:);
% count the number of occurances of for each words when Y=0/Y=1
model(:,1) =  sum(Xtrain0, 1);
model(:,2) =  sum(Xtrain1, 1);
model = model + 1;   %smoothing
%count conditional probability
sum0 = sum(model(:,1));
sum1 = sum(model(:,2));
model(:,1) = log10(model(:,1) / sum0);
model(:,2) = log10(model(:,2) / sum1);
model = [[P0,P1];model];