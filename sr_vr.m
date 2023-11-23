function [sr_vr_knn, sr_vr_tree ,sr_vr_tb] = sr_vr(M,q)

temp1 = zeros(6,5);
temp2 = zeros(6,5);
temp3 = zeros(6,5);

for i = 0:size(M,1)/15-1
temp1(i+1,:) = M(15*i+1:15*i+5,q)';
end
sr_vr_knn = mean(temp1);

for i = 0:size(M,1)/15-1
temp2(i+1,:) = M(15*i+6:15*i+10,q);
end
sr_vr_tree = mean(temp2);

for i = 0:size(M,1)/15-1
temp3(i+1,:) = M(15*i+11:15*i+15,q);
end
sr_vr_tb = mean(temp3);

end

