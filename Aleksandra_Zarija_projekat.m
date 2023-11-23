l_bm = pravljenje('bm','*L_.wav.txt',1);
n_bm = pravljenje('bm','*N_.wav.txt',2);
r_bm = pravljenje('bm','*R_.wav.txt',3);
s_bm = pravljenje('bm','*S_.wav.txt',4);
t_bm = pravljenje('bm','*T_.wav.txt',5);

l_mm = pravljenje('mm','*L_.wav.txt',1);
n_mm = pravljenje('mm','*N_.wav.txt',2);
r_mm = pravljenje('mm','*R_.wav.txt',3);
s_mm = pravljenje('mm','*S_.wav.txt',4);
t_mm = pravljenje('mm','*T_.wav.txt',5);

l_mv = pravljenje('mv','*L_.wav.txt',1);
n_mv = pravljenje('mv','*N_.wav.txt',2);
r_mv = pravljenje('mv','*R_.wav.txt',3);
s_mv = pravljenje('mv','*S_.wav.txt',4);
t_mv = pravljenje('mv','*T_.wav.txt',5);

l_ok = pravljenje('ok','*L_.wav.txt',1);
n_ok = pravljenje('ok','*N_.wav.txt',2);
r_ok = pravljenje('ok','*R_.wav.txt',3);
s_ok = pravljenje('ok','*S_.wav.txt',4);
t_ok = pravljenje('ok','*T_.wav.txt',5);

l_sk = pravljenje('sk','*L_.wav.txt',1);
n_sk = pravljenje('sk','*N_.wav.txt',2);
r_sk = pravljenje('sk','*R_.wav.txt',3);
s_sk = pravljenje('sk','*S_.wav.txt',4);
t_sk = pravljenje('sk','*T_.wav.txt',5);

l_sz = pravljenje('sz','*L_.wav.txt',1);
n_sz = pravljenje('sz','*N_.wav.txt',2);
r_sz = pravljenje('sz','*R_.wav.txt',3);
s_sz = pravljenje('sz','*S_.wav.txt',4);
t_sz = pravljenje('sz','*T_.wav.txt',5);

D_bm = [l_bm ; n_bm ; r_bm ; s_bm ; t_bm ];
D_mm = [l_mm ; n_mm ; r_mm ; s_mm ; t_mm ];
D_mv = [l_mv ; n_mv ; r_mv ; s_mv ; t_mv ];
D_ok = [l_ok ; n_ok ; r_ok ; s_ok ; t_ok ];
D_sk = [l_sk ; n_sk ; r_sk ; s_sk ; t_sk ];
D_sz = [l_sz ; n_sz ; r_sz ; s_sz ; t_sz ];

D_bm1 = [D_bm ones(size(D_bm,1),1)];
D_mm2 = [D_mm 2*ones(size(D_bm,1),1)];
D_mv3 = [D_mv 3*ones(size(D_bm,1),1)];
D_ok4 = [D_ok 4*ones(size(D_bm,1),1)];
D_sk5 = [D_sk 5*ones(size(D_bm,1),1)];
D_sz6 = [D_sz 6*ones(size(D_bm,1),1)];

D = [D_bm1; D_mm2 ; D_mv3 ; D_ok4 ; D_sk5 ; D_sz6 ];

data_train = D(:,1:6);
lab_train = D(:,7);
osoba_train = D(:,8);

C_knn_final = zeros(5);
C_tree_final = zeros(5);
C_tb_final = zeros(5);

C_big_final = [];
Z = [];

c = cvpartition(lab_train,'KFold',5);    

for i = 1:6
    
    test_org = data_train(osoba_train == i,:);
    test_lab = lab_train(osoba_train == i,:);

    tr_org = data_train(osoba_train ~= i,:);
    tr_lab = lab_train(osoba_train ~= i,:);

%kNN

    model_knn = fitcknn(tr_org, tr_lab,'NumNeighbors',3, 'Distance', 'euclidean', 'Standardize', 1);
    pred_lab_knn = predict(model_knn, test_org);
    [C_knn,~] = confusionmat(test_lab, pred_lab_knn);
    C_knn_final = C_knn_final + C_knn;

%stablo

    model_tree = fitctree(tr_org, tr_lab,'MaxNumSplits',11);
    pred_lab_tree = predict(model_tree, test_org);
    [C_tree,~] = confusionmat(test_lab, pred_lab_tree);
    C_tree_final = C_tree_final + C_tree;
    
%     view(model_tree,'Mode','graph');
    
%TreeBagger

    model_tb = TreeBagger(12,tr_org,tr_lab,'OOBPrediction','On','Method','classification');
    pred_lab_tb = predict(model_tb, test_org);
    [C_tb,~] = confusionmat(test_lab, str2double(pred_lab_tb));
    C_tb_final = C_tb_final + C_tb;
    
%     figure;
%     oobErrorBaggedEnsemble = oobError(model_tb);
%     plot(oobErrorBaggedEnsemble)
%     xlabel 'Number of grown trees';
%     ylabel 'Out-of-bag classification error';

%     C_big_final = [C_big_final; C_knn_final; C_tree_final; C_tb_final];
end
 
% %mere validacije
% for i = 0:size(C_big_final,1)/5-1
% 
% C_big = C_big_final(5*i+1:5*i+5,:);
% M(5*i+1:5*i+5,:) = mere(C_big);
% 
% end
%kNN
%%ljutnja
prec_knn_l = C_knn_final(1,1)/sum(C_knn_final(1,:));
acc_knn_l = (C_knn_final(1,1) + sum(sum(C_knn_final(2:5,2:5))))/sum(sum(C_knn_final));
rec_knn_l = C_knn_final(1,1)/sum(C_knn_final(:,1));
spec_knn_l = sum(sum(C_knn_final(2:5,2:5)))/sum(sum(C_knn_final(2:5,:)));
F1_knn_l = 2/(1/rec_knn_l + 1/prec_knn_l);

%%neutralno
prec_knn_n = C_knn_final(2,2)/sum(C_knn_final(2,:));
acc_knn_n = (C_knn_final(1,1) + C_knn_final(2,2) + sum(C_knn_final(1,3:5)) + sum(C_knn_final(3:5,1)) + sum(sum(C_knn_final(3:5,3:5))))/sum(sum(C_knn_final));
rec_knn_n = C_knn_final(2,2)/sum(C_knn_final(:,2));
spec_knn_n = (C_knn_final(1,1) + sum(C_knn_final(1,3:5)) + sum(C_knn_final(3:5,1)) + sum(sum(C_knn_final(3:5,3:5))))/(sum(C_knn_final(:,1)) + sum(sum(C_knn_final(:,3:5))));
F1_knn_n = 2/(1/rec_knn_n + 1/prec_knn_n);

%%radost
prec_knn_r = C_knn_final(3,3)/sum(C_knn_final(3,:));
acc_knn_r = (C_knn_final(3,3) + sum(sum(C_knn_final(1:2,1:2))) + sum(sum(C_knn_final(1:2,4:5))) + sum(sum(C_knn_final(4:5,1:2))) + sum(sum(C_knn_final(4:5,4:5))))/sum(sum(C_knn_final));
rec_knn_r = C_knn_final(3,3)/sum(C_knn_final(:,3));
spec_knn_r = (sum(sum(C_knn_final(1:2,1:2))) + sum(sum(C_knn_final(1:2,4:5))) + sum(sum(C_knn_final(4:5,1:2))) + sum(sum(C_knn_final(4:5,4:5))))/(sum(sum(C_knn_final(:,1:2))) + sum(sum(C_knn_final(:,4:5))));
F1_knn_r = 2/(1/rec_knn_r + 1/prec_knn_r);

%%strah
prec_knn_s = C_knn_final(4,4)/sum(C_knn_final(4,:));
acc_knn_s = (C_knn_final(4,4) + sum(sum(C_knn_final(1:3,1:3))) + sum(C_knn_final(1:3,5)) + sum(C_knn_final(5,1:3)) + C_knn_final(5,5))/sum(sum(C_knn_final));
rec_knn_s = C_knn_final(4,4)/sum(C_knn_final(:,4));
spec_knn_s = (sum(sum(C_knn_final(1:3,1:3))) + sum(C_knn_final(1:3,5)) + sum(C_knn_final(5,1:3)) + C_knn_final(5,5))/(sum(sum(C_knn_final(1:5,1:3)))+sum(C_knn_final(1:5,5)));
F1_knn_s = 2/(1/rec_knn_s + 1/prec_knn_s);

%%tuga
prec_knn_t = C_knn_final(5,5)/sum(C_knn_final(5,:));
acc_knn_t = (C_knn_final(5,5) + sum(sum(C_knn_final(1:4,1:4))))/sum(sum(C_knn_final));
rec_knn_t = C_knn_final(5,5)/sum(C_knn_final(:,5));
spec_knn_t = sum(sum(C_knn_final(1:4,1:4)))/sum(sum(C_knn_final(1:5,1:4)));
F1_knn_t = 2/(1/rec_knn_t + 1/prec_knn_t);

knn_matrix = [prec_knn_l prec_knn_n prec_knn_r prec_knn_s prec_knn_t; acc_knn_l acc_knn_n acc_knn_r acc_knn_s acc_knn_t;rec_knn_l rec_knn_n rec_knn_r rec_knn_s rec_knn_t; spec_knn_l spec_knn_n spec_knn_r spec_knn_s spec_knn_t; F1_knn_l F1_knn_n F1_knn_r F1_knn_s F1_knn_t];
%stablo
%%ljutnja
prec_tree_l = C_tree_final(1,1)/sum(C_tree_final(1,:));
acc_tree_l = (C_tree_final(1,1) + sum(sum(C_tree_final(2:5,2:5))))/sum(sum(C_tree_final));
rec_tree_l = C_tree_final(1,1)/sum(C_tree_final(:,1));
spec_tree_l = sum(sum(C_tree_final(2:5,2:5)))/sum(sum(C_tree_final(2:5,:)));
F1_tree_l = 2/(1/rec_tree_l + 1/prec_tree_l);

%%neutralno
prec_tree_n = C_tree_final(2,2)/sum(C_tree_final(2,:));
acc_tree_n = (C_tree_final(2,2) + C_tree_final(1,1) + sum(C_tree_final(1,3:5)) + sum(C_tree_final(3:5,1)) + sum(sum(C_tree_final(3:5,3:5))))/sum(sum(C_tree_final));
rec_tree_n = C_tree_final(2,2)/sum(C_tree_final(:,2));
spec_tree_n = (C_tree_final(1,1) + sum(C_tree_final(1,3:5)) + sum(C_tree_final(3:5,1)) + sum(sum(C_tree_final(3:5,3:5))))/(sum(C_tree_final(:,1)) + sum(sum(C_tree_final(:,3:5))));
F1_tree_n = 2/(1/rec_tree_n + 1/prec_tree_n);

%%radost
prec_tree_r = C_tree_final(3,3)/sum(C_tree_final(3,:));
acc_tree_r = (C_tree_final(3,3) + sum(sum(C_tree_final(1:2,1:2))) + sum(sum(C_tree_final(1:2,4:5))) + sum(sum(C_tree_final(4:5,1:2))) + sum(sum(C_tree_final(4:5,4:5))))/sum(sum(C_tree_final));
rec_tree_r = C_tree_final(3,3)/sum(C_tree_final(:,3));
spec_tree_r = (sum(sum(C_tree_final(1:2,1:2))) + sum(sum(C_tree_final(1:2,4:5))) + sum(sum(C_tree_final(4:5,1:2))) + sum(sum(C_tree_final(4:5,4:5))))/(sum(sum(C_tree_final(:,1:2))) + sum(sum(C_tree_final(:,4:5))));
F1_tree_r = 2/(1/rec_tree_r + 1/prec_tree_r);

%%strah
prec_tree_s = C_tree_final(4,4)/sum(C_tree_final(4,:));
acc_tree_s = (C_tree_final(4,4) + sum(sum(C_tree_final(1:3,1:3))) + sum(C_tree_final(1:3,5)) + sum(C_tree_final(5,1:3)) + C_tree_final(5,5))/sum(sum(C_tree_final));
rec_tree_s = C_tree_final(4,4)/sum(C_tree_final(:,4));
spec_tree_s = (sum(sum(C_tree_final(1:3,1:3))) + sum(C_tree_final(1:3,5)) + sum(C_tree_final(5,1:3)) + C_tree_final(5,5))/(sum(sum(C_tree_final(1:5,1:3)))+sum(C_tree_final(1:5,5)));
F1_tree_s = 2/(1/rec_tree_s + 1/prec_tree_s);

%%tuga
prec_tree_t = C_tree_final(5,5)/sum(C_tree_final(5,:));
acc_tree_t = (C_tree_final(5,5) + sum(sum(C_tree_final(1:4,1:4))))/sum(sum(C_tree_final));
rec_tree_t = C_tree_final(5,5)/sum(C_tree_final(:,5));
spec_tree_t = sum(sum(C_tree_final(1:4,1:4)))/sum(sum(C_tree_final(1:5,1:4)));
F1_tree_t = 2/(1/rec_tree_t + 1/prec_tree_t);

tree_matrix = [prec_tree_l prec_tree_n prec_tree_r prec_tree_s prec_tree_t; acc_tree_l acc_tree_n acc_tree_r acc_tree_s acc_tree_t; rec_tree_l rec_tree_n rec_tree_r rec_tree_s rec_tree_t; spec_tree_l spec_tree_n spec_tree_r spec_tree_s spec_tree_t; F1_tree_l F1_tree_n F1_tree_r F1_tree_s F1_tree_t];

%TreeBagger
%%ljutnja
prec_tb_l = C_tb_final(1,1)/sum(C_tb_final(1,:));
acc_tb_l = (C_tb_final(1,1) + sum(sum(C_tb_final(2:5,2:5))))/sum(sum(C_tb_final)); %?
rec_tb_l = C_tb_final(1,1)/sum(C_tb_final(:,1));
spec_tb_l = sum(sum(C_tb_final(2:5,2:5)))/sum(sum(C_tb_final(2:5,:)));
F1_tb_l = 2/(1/rec_tb_l + 1/prec_tb_l);

%%neutralno
prec_tb_n = C_tb_final(2,2)/sum(C_tb_final(2,:));
acc_tb_n = (C_tb_final(2,2) + (C_tb_final(1,1) + sum(C_tb_final(1,3:5)) + sum(C_tb_final(3:5,1)) + sum(sum(C_tb_final(3:5,3:5)))))/sum(sum(C_tb_final));
rec_tb_n = C_tb_final(2,2)/sum(C_tb_final(:,2));
spec_tb_n = (C_tb_final(1,1) + sum(C_tb_final(1,3:5)) + sum(C_tb_final(3:5,1)) + sum(sum(C_tb_final(3:5,3:5))))/(sum(C_tb_final(:,1)) + sum(sum(C_tb_final(:,3:5))));
F1_tb_n = 2/(1/rec_tb_n + 1/prec_tb_n);

%%radost
prec_tb_r = C_tb_final(3,3)/sum(C_tb_final(3,:));
acc_tb_r = (C_tb_final(3,3) + sum(sum(C_tb_final(1:2,1:2))) + sum(sum(C_tb_final(1:2,4:5))) + sum(sum(C_tb_final(4:5,1:2))) + sum(sum(C_tb_final(4:5,4:5))))/sum(sum(C_tb_final));
rec_tb_r = C_tb_final(3,3)/sum(C_tb_final(:,3));
spec_tb_r = (sum(sum(C_tb_final(1:2,1:2))) + sum(sum(C_tb_final(1:2,4:5))) + sum(sum(C_tb_final(4:5,1:2))) + sum(sum(C_tb_final(4:5,4:5))))/(sum(sum(C_tb_final(:,1:2))) + sum(sum(C_tb_final(:,4:5))));
F1_tb_r = 2/(1/rec_tb_r + 1/prec_tb_r);

%%strah
prec_tb_s = C_tb_final(4,4)/sum(C_tb_final(4,:));
acc_tb_s = (C_tb_final(4,4) + sum(sum(C_tb_final(1:3,1:3))) + sum(C_tb_final(1:3,5)) + sum(C_tb_final(5,1:3)) + C_tb_final(5,5))/sum(sum(C_tb_final));
rec_tb_s = C_tb_final(4,4)/sum(C_tb_final(:,4));
spec_tb_s = (sum(sum(C_tb_final(1:3,1:3))) + sum(C_tb_final(1:3,5)) + sum(C_tb_final(5,1:3)) + C_tb_final(5,5))/(sum(sum(C_tb_final(1:5,1:3)))+sum(C_tb_final(1:5,5)));
F1_tb_s = 2/(1/rec_tb_s + 1/prec_tb_s);

%%tuga
prec_tb_t = C_tb_final(5,5)/sum(C_tb_final(5,:));
acc_tb_t = (C_tb_final(5,5) + sum(sum(C_tb_final(1:4,1:4))))/sum(sum(C_tb_final));
rec_tb_t = C_tb_final(5,5)/sum(C_tb_final(:,5));
spec_tb_t = sum(sum(C_tb_final(1:4,1:4)))/sum(sum(C_tb_final(1:5,1:4)));
F1_tb_t = 2/(1/rec_tb_t + 1/prec_tb_t);
 
tb_matrix = [prec_tb_l prec_tb_n prec_tb_r prec_tb_s prec_tb_t; acc_tb_l acc_tb_n acc_tb_r acc_tb_s acc_tb_t; rec_tb_l rec_tb_n rec_tb_r rec_tb_s rec_tb_t; spec_tb_l spec_tb_n spec_tb_r spec_tb_s spec_tb_t; F1_tb_l F1_tb_n F1_tb_r F1_tb_s F1_tb_t];

%stopa prepoznavanja
rec_rate_knn = sum(diag(C_knn_final))/sum(sum(C_knn_final));
rec_rate_tree = sum(diag(C_tree_final))/sum(sum(C_tree_final));
rec_rate_tb = sum(diag(C_tb_final))/sum(sum(C_tb_final));

%grafici
c = categorical({'kNN','Tree','TreeBagger'});

bar(c,[knn_matrix(:,1)'; tree_matrix(:,1)'; tb_matrix(:,1)'])
title('Srednje vrednosti mera za ljutnju');

bar(c,[knn_matrix(:,2)'; tree_matrix(:,2)'; tb_matrix(:,2)'])
title('Srednje vrednosti mera za neutralno stanje');

bar(c,[knn_matrix(:,3)'; tree_matrix(:,3)'; tb_matrix(:,3)'])
title('Srednje vrednosti mera za radost');

bar(c,[knn_matrix(:,4)'; tree_matrix(:,4)'; tb_matrix(:,4)'])
title('Srednje vrednosti mera za strah');

bar(c,[knn_matrix(:,5)'; tree_matrix(:,5)'; tb_matrix(:,5)'])
title('Srednje vrednosti mera za tugu');

bar(c,[rec_rate_knn ; rec_rate_tree; rec_rate_tb])
title('Stopa prepoznavanja');