
clear all
clc


Dataname = 'CASIA_multiview_DC_3'


load(Dataname);

disp('Randomly select the training data........')
sele_num = 4;
nnClass = length(unique(gnd));  % The number of classes;
num_Class = [];
for i = 1:nnClass
  num_Class = [num_Class length(find(gnd==i))]; %The number of samples of each class
end

iv=length(fea);
% iv=1;
for i=1:iv
    Train_Ma{i} = [];   
    Test_Ma{i} = []; 
    fea{i} = double(fea{i}');
end

Train_Lab = [];
Test_Lab = [];

for j = 1:nnClass    
    idx      = find(gnd==j);
    randIdx  = randperm(num_Class(j));
    Train_Lab = [Train_Lab;gnd(idx(randIdx(1:sele_num)))];
    Test_Lab = [Test_Lab;gnd(idx(randIdx(sele_num+1:num_Class(j))))];
    for ii=1:iv
        Train_Ma{ii} = [Train_Ma{ii}; fea{ii}(idx(randIdx(1:sele_num)),:)];            % select select_num samples per class for training       
        Test_Ma{ii}  = [Test_Ma{ii};fea{ii}(idx(randIdx(sele_num+1:num_Class(j))),:)];  % select remaining samples per class for test        
    end
end

for i=1:iv
    Train_Ma1{i} = Train_Ma{i}';                       % transform to a sample per column
    Train_Ma1{i} = Train_Ma1{i}./repmat(sqrt(sum(Train_Ma1{i}.^2)),[size(Train_Ma1{i},1) 1]);
    Test_Ma1{i}  = Test_Ma{i}';
    Test_Ma1{i}  = Test_Ma1{i}./repmat(sqrt(sum(Test_Ma1{i}.^2)),[size(Test_Ma1{i},1) 1]);  % -------------
end

label = unique(Train_Lab);
Y = bsxfun(@eq, Train_Lab, label');
Y = double(Y)';

X = Train_Ma1;
n = length(Train_Lab);
Z = zeros(n,n);
for i = 1:n
    v = Z (:,i);  
    v((fix((i-1)/sele_num)*sele_num+1):(fix((i-1)/sele_num)*sele_num+sele_num)) = 1;
    Z(:,i) = v;
end


opts.miu = 1e-8;
opts.rho = 1.01;
opts.lambda1 = 1e-2;%1e-2
opts.lambda2 = 5e-5;%5e-1
opts.nnClass = nnClass;
opts.maxIter =1;

disp('Training start........')
[Q,E] = DC_MDPR(X, Y, Z, Train_Lab, opts);%double-cohesion learning multiview palmprint recognition


Train_Maa = zeros(size(Y));
Test_Maa = zeros(nnClass,length(Test_Lab));


for i = 1:iv
    Train_Maa = Train_Maa+Q{i}*X{i};
    Test_Maa  = Test_Maa+Q{i}*Test_Ma1{i};
end
disp('Test start........')
[class_test] = knnclassify(Test_Maa', Train_Maa', Train_Lab,1,'euclidean','nearest');
rate_acc = sum(Test_Lab == class_test)/length(Test_Lab)*100
