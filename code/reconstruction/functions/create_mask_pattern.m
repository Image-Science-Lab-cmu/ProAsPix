function mask = create_mask_pattern(assort_index, thres)

mask = 0*assort_index;

list = [-1 0; 0 -1; 1 0; 0 1];

absdiff = 0;
siz = size(assort_index);

for ee=1:size(list, 1)
    
    absdiff = max(absdiff, abs( assort_index(2:end-1, 2:end-1)-assort_index(list(ee,1)+(2:end-1), list(ee,2)+(2:end-1))));
end

mask(2:end-1, 2:end-1) = absdiff;
mask = mask < thres;
end