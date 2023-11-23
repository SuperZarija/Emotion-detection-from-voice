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

%analiza podataka
boxplot([l_bm(:,[1 2 4 5]); l_mm(:,[1 2 4 5]); l_mv(:,[1 2 4 5]); l_ok(:,[1 2 4 5]); l_sk(:,[1 2 4 5]); l_sz(:,[1 2 4 5])]);
boxplot([n_bm(:,[1 2 4 5]); l_mm(:,[1 2 4 5]); l_mv(:,[1 2 4 5]); l_ok(:,[1 2 4 5]); l_sk(:,[1 2 4 5]); l_sz(:,[1 2 4 5])]);
boxplot([r_bm(:,[1 2 4 5]); l_mm(:,[1 2 4 5]); l_mv(:,[1 2 4 5]); l_ok(:,[1 2 4 5]); l_sk(:,[1 2 4 5]); l_sz(:,[1 2 4 5])]);
boxplot([s_bm(:,[1 2 4 5]); l_mm(:,[1 2 4 5]); l_mv(:,[1 2 4 5]); l_ok(:,[1 2 4 5]); l_sk(:,[1 2 4 5]); l_sz(:,[1 2 4 5])]);
boxplot([t_bm(:,[1 2 4 5]); l_mm(:,[1 2 4 5]); l_mv(:,[1 2 4 5]); l_ok(:,[1 2 4 5]); l_sk(:,[1 2 4 5]); l_sz(:,[1 2 4 5])]);

[cnt1,L] = pct_cnt(D,1);
[cnt2,N] = pct_cnt(D,2);
[cnt3,R] = pct_cnt(D,3);
[cnt4,S] = pct_cnt(D,4);
[cnt5,T] = pct_cnt(D,5);

%klasifikacija
data_train = D(:,1:6);
lab_train = D(:,7);
osoba_train = D(:,8);

C_knn_final = zeros(5);
C_tree_final = zeros(5);
C_tb_final = zeros(5);

C_big_final = [];
M = [];

for j = 1:6
    
    data_train_osoba = data_train(osoba_train == j,:);
    lab_train_osoba = lab_train(osoba_train == j,:);
    
    c = cvpartition(lab_train_osoba,'KFold',10);
    
    for i = 1:5
        
    test_org = data_train_osoba(c.test(i),:);
    test_lab = lab_train_osoba(c.test(i),:);

    tr_org = data_train_osoba(c.training(i),:);
    tr_lab = lab_train_osoba(c.training(i),:);
    

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
    
    %view(model_tree,'Mode','graph');
    
%TreeBagger

    model_tb = TreeBagger(50,tr_org,tr_lab,'OOBPrediction','On','Method','classification');
    pred_lab_tb = predict(model_tb, test_org);
    [C_tb,~] = confusionmat(test_lab, str2double(pred_lab_tb));
    C_tb_final = C_tb_final + C_tb;
    
%     figure;
%     oobErrorBaggedEnsemble = oobError(model_tb);
%     plot(oobErrorBaggedEnsemble)
%     xlabel 'Number of grown trees';
%     ylabel 'Out-of-bag classification error';
    end
    
    C_big_final = [C_big_final; C_knn_final; C_tree_final; C_tb_final];
end

%mere validacije
for i = 0:size(C_big_final,1)/5-1

C_big = C_big_final(5*i+1:5*i+5,:);
M(5*i+1:5*i+5,:) = mere(C_big);

end

%srednje vrednosti mera za svaku emociju
[l_sr_vr_knn , l_sr_vr_tree, l_sr_vr_tb] = sr_vr(M,1);
[n_sr_vr_knn , n_sr_vr_tree, n_sr_vr_tb] = sr_vr(M,2);
[r_sr_vr_knn , r_sr_vr_tree, r_sr_vr_tb] = sr_vr(M,3);
[s_sr_vr_knn , s_sr_vr_tree, s_sr_vr_tb] = sr_vr(M,4);
[t_sr_vr_knn , t_sr_vr_tree, t_sr_vr_tb] = sr_vr(M,5);

%stopa prepoznavanja
rec_rate_knn = sum(diag(C_knn_final))/sum(sum(C_knn_final));
rec_rate_tree = sum(diag(C_tree_final))/sum(sum(C_tree_final));
rec_rate_tb = sum(diag(C_tb_final))/sum(sum(C_tb_final));

%grafici
c = categorical({'kNN','Tree','TreeBagger'});

bar(c,[l_sr_vr_knn ; l_sr_vr_tree; l_sr_vr_tb])
title('Srednje vrednosti mera za ljutnju');

bar(c,[n_sr_vr_knn ; n_sr_vr_tree; n_sr_vr_tb])
title('Srednje vrednosti mera za neutralno stanje');

bar(c,[r_sr_vr_knn ; r_sr_vr_tree; r_sr_vr_tb])
title('Srednje vrednosti mera za radost');

bar(c,[s_sr_vr_knn ; s_sr_vr_tree; s_sr_vr_tb])
title('Srednje vrednosti mera za strah');

bar(c,[t_sr_vr_knn ; t_sr_vr_tree; t_sr_vr_tb])
title('Srednje vrednosti mera za tugu');

bar(c,[rec_rate_knn ; rec_rate_tree; rec_rate_tb])
title('Stopa prepoznavanja');