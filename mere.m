function mere_validacije = mere(C)

%ljutnja
prec_l = C(1,1)/sum(C(1,:));
acc_l = (C(1,1) + sum(sum(C(2:5,2:5))))/sum(sum(C));
rec_l = C(1,1)/sum(C(:,1));
spec_l = sum(sum(C(2:5,2:5)))/sum(sum(C(2:5,:)));
F1_l = 2/(1/rec_l + 1/prec_l);

%%neutralno
prec_n = C(2,2)/sum(C(2,:));
acc_n = (C(1,1) + C(2,2) + sum(C(1,3:5)) + sum(C(3:5,1)) + sum(sum(C(3:5,3:5))))/sum(sum(C));
rec_n = C(2,2)/sum(C(:,2));
spec_n = (C(1,1) + sum(C(1,3:5)) + sum(C(3:5,1)) + sum(sum(C(3:5,3:5))))/(sum(C(:,1)) + sum(sum(C(:,3:5))));
F1_n = 2/(1/rec_n + 1/prec_n);

%%radost
prec_r = C(3,3)/sum(C(3,:));
acc_r = (C(3,3) + sum(sum(C(1:2,1:2))) + sum(sum(C(1:2,4:5))) + sum(sum(C(4:5,1:2))) + sum(sum(C(4:5,4:5))))/sum(sum(C));
rec_r = C(3,3)/sum(C(:,3));
spec_r = (sum(sum(C(1:2,1:2))) + sum(sum(C(1:2,4:5))) + sum(sum(C(4:5,1:2))) + sum(sum(C(4:5,4:5))))/(sum(sum(C(:,1:2))) + sum(sum(C(:,4:5))));
F1_r = 2/(1/rec_r + 1/prec_r);

%%strah
prec_s = C(4,4)/sum(C(4,:));
acc_s = (C(4,4) + sum(sum(C(1:3,1:3))) + sum(C(1:3,5)) + sum(C(5,1:3)) + C(5,5))/sum(sum(C));
rec_s = C(4,4)/sum(C(:,4));
spec_s = (sum(sum(C(1:3,1:3))) + sum(C(1:3,5)) + sum(C(5,1:3)) + C(5,5))/(sum(sum(C(1:5,1:3))) + sum(C(1:5,5)));
F1_s = 2/(1/rec_s + 1/prec_s);

%%tuga
prec_t = C(5,5)/sum(C(5,:));
acc_t = (C(5,5) + sum(sum(C(1:4,1:4))))/sum(sum(C));
rec_t = C(5,5)/sum(C(:,5));
spec_t = sum(sum(C(1:4,1:4)))/sum(sum(C(1:5,1:4)));
F1_t = 2/(1/rec_t + 1/prec_t);

prec = [prec_l prec_n prec_r prec_s prec_t];
acc = [acc_l acc_n acc_r acc_s acc_t];
rec = [rec_l rec_n rec_r rec_s rec_t];
spec = [spec_l spec_n spec_r spec_s spec_t];
F1 = [F1_l F1_n F1_r F1_s F1_t];

mere_validacije = [prec ; acc; rec; spec; F1];
end

