function Pred_nb = nb_test(model, Xtest)
row = size(Xtest,1);
X = [ones(row, 1), Xtest];
%size(X), size(model)
prediction = X * model;
Pred_nb=zeros(row,1);
for i = 1:row
    if prediction(i,1)<prediction(i,2)
        Pred_nb(i,1)=1;
    end
end