function matrica = pravljenje(folder, naziv, lab)
cd(folder);

matrica = zeros(60,6); %pr: folder bm
d = dir(naziv); %pr: '*S_.wav.txt'
for i = 1:size(d,1)
    p = load(d(i).name);
    p = p(find(p));
    
    if sum(diff(p))>0.5 %pitch contur trends
        pct = 1; %'inclines'
    elseif sum(diff(p))<-0.5
        pct = -1; %'declines'
    elseif abs(sum(diff(p)))<0.5
        pct = 0; %'stagnate'
    end
    
    temp = [mean(p) std(p) var(p) iqr(p) max(p)-min(p) pct]; 
    matrica(i,:) = temp;
end

matrica(:,end+1) = lab;

cd ../
