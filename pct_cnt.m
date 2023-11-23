function [cnt,I] = pct_cnt(D,p)

cnt_up = 0;
cnt_down = 0;
cnt_stag = 0;
temp = [];

for i = 1:size(D,1)
    if(D(i,7) == p)
        temp = D(i,6);
    end


    if temp == 1
        cnt_up = cnt_up+1;
    elseif temp == -1
        cnt_down = cnt_down + 1;
    else
        cnt_stag = cnt_stag + 1;
    end
end

[cnt,I] = max([cnt_up cnt_down cnt_stag]);
    
end

